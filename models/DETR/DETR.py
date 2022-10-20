import os
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from dataset import VOC2007
from losses.Hungarian import hungarian_loss, match_loss

weight_path = '../../weights/detr-voc2007.pth'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dims=256, nheads=8, num_encoder_layer=6, num_decoder_layer=6):
        super(DETR, self).__init__()
        self.num_queries = 100
        self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        del self.backbone.fc
        del self.backbone.avgpool

        self.backbone_channels = 2048
        self.conv1 = nn.Conv2d(self.backbone_channels, hidden_dims, 1)
        self.bn = nn.BatchNorm2d(hidden_dims)

        # todo: use pytorch implementation at the present, we should write this ourselves.
        self.transformer = nn.Transformer(hidden_dims, nhead=nheads, num_encoder_layers=num_encoder_layer,
                                          num_decoder_layers=num_decoder_layer)

        # FFN to predict the bbox.
        self.linear_class = nn.Linear(hidden_dims, num_classes + 1)
        self.linear_bbox = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims * 2),
            nn.ReLU(),
            nn.Linear(hidden_dims * 2, 4)
        )

        # object queries, for now the number of object query is  ->   100
        # self.object_queries = torch.randint(low=1, high=100, size=(self.num_queries,1))
        self.object_queries = nn.Parameter(torch.rand(self.num_queries, hidden_dims))
        self.query_embed = nn.Embedding(self.num_queries+1, hidden_dims)

        # spatial positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dims // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dims // 2))

    def forward(self, x=torch.rand(1, 3, 224, 224)):
        # use resnet50 to extract features. (b, c, h, w): (b, 2048, 7, 7)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # use conv1 to make channels 2048 to size of hidden_dims=256 (b, 256, 7, 7)
        h = self.conv1(x)
        h = self.bn(h)
        # make positional encoding, shape: (featuremap_h * featuremap_w, 1, hidden_dims)
        H, W = h.shape[-2:]
        col = self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1)
        row = self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)
        positional_encoding = torch.cat((col, row), dim=-1).flatten(0, 1).unsqueeze(1)
        tgt = self.object_queries.unsqueeze(1)
        h = self.transformer(positional_encoding + 0.1 * h.flatten(2).permute(2, 0, 1),
                             tgt) \
            .transpose(0, 1)
        pred_class = self.linear_class(h)
        pred_bbox = self.linear_bbox(h).sigmoid()
        # output shape: torch.Size([1, 100, 21]) torch.Size([1, 100, 4])
        return {'pred_class': pred_class, 'pred_bbox': pred_bbox}
