import time

import torch
import torchvision
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from dataset import VOC2007
from losses.Hungarian import hungarian_loss


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dims=256, nheads=8, num_encoder_layer=6, num_decoder_layer=6):
        super(DETR, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        del self.backbone.fc
        del self.backbone.avgpool

        self.backbone_channels = 2048
        self.conv1 = nn.Conv2d(self.backbone_channels, hidden_dims, 1)

        # todo: use pytorch implementation at the present, we should write this ourselves.
        self.transformer = nn.Transformer(hidden_dims, nhead=nheads, num_encoder_layers=num_encoder_layer,
                                          num_decoder_layers=num_decoder_layer)

        # FFN to predict the bbox.
        self.linear_class = nn.Linear(hidden_dims, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dims, 4)

        # object queries, for now the number of object query is  ->   100
        self.object_queries = nn.Parameter(torch.rand(100, hidden_dims))

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
        # make positional encoding, shape: (featuremap_h * featuremap_w, 1, hidden_dims)
        H, W = h.shape[-2:]
        col = self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1)
        row = self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)
        positional_encoding = torch.cat((col, row), dim=-1).flatten(0, 1).unsqueeze(1)
        h = self.transformer(positional_encoding + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.object_queries.unsqueeze(1)) \
            .transpose(0, 1)
        pred_class = self.linear_class(h)
        pred_bbox = self.linear_bbox(h)
        # output shape: torch.Size([1, 100, 21]) torch.Size([1, 100, 4])
        return {'pred_class': pred_class, 'pred_bbox': pred_bbox}


def train(batch_size=1, epoches=100, learning_rate=0.01, weight_decay=1e-5):
    # init device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # prepare the datasets
    dataset_root_path = 'D:\\code\\python\\datasets\\VOCdevkit\\VOC2007'
    train_data = VOC2007(root_path=dataset_root_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    # network
    net = DETR(num_classes=20)
    net = net.to(device)

    # optimizer
    optimizer = SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # loss function use hungarian loss to criterion
    criterian = hungarian_loss

    for epoch in range(1, epoches + 1):
        start_time = time.time()
        image_num = 0
        total_loss = 0
        for i, batch in enumerate(train_loader):
            image_num += 1
            image = batch[0][0]
            height, width = batch[0][2], batch[0][3]
            image = image.unsqueeze(0)
            gt_boxes = torch.tensor(batch[0][1])
            output = net(image)
            pred_class, pred_bbox = output['pred_class'], output['pred_bbox']
            # until this point
            # gtboxes: (gt_box_number, clsx1y1x2y2)
            # pred_class: (batch_size, objectquery, 21(class number))
            # pred_bbox: (batch_size, objectquery, x1y1x2y2)
            break


if __name__ == '__main__':
    # net = DETR(num_classes=20)
    # print(net()['pred_class'].shape, net()['pred_bbox'].shape)
    train()
