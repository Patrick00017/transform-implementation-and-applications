import torch
from models.yolov3.model import Yolov3
import os
import numpy as np
import xml.etree.ElementTree as ET
from models.yolov3.dataset import get_train_dataset
from torch.utils.data import DataLoader
from tools.bbox import get_objectness_label

batch_size = 16
num_classes = 7  # include playground
num_anchor = 3
epoches = 100


def train():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_dataset, classname2idx, idx2classname = get_train_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    net = Yolov3(num_classes=num_classes, num_anchor_per_pixel=num_anchor)
    net = net.to(device)

    print('start training...')
    for epoch in range(1, epoches + 1):
        for i, batch in enumerate(train_loader):
            imgs, gt_boxes, gt_labels, im_shape = batch
            imgs = imgs.to(device)
            p0, p1, p2 = net(imgs)
            # shape:
            # torch.Size([16, 60, 7, 7]) torch.Size([16, 60, 14, 14]) torch.Size([16, 60, 28, 28])
            downsample = 32
            anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
            for i, layer in enumerate([p0, p1, p2]):
                layer_anchor = anchors[i]
                layer_shape = layer.shape
                layer = torch.reshape(layer, (batch_size, num_anchor, -1, layer_shape[-2], layer_shape[-1]))
                # print(layer.shape)
                # torch.Size([16, 3, 12, 7, 7])
                # torch.Size([16, 3, 12, 14, 14])
                # torch.Size([16, 3, 12, 28, 28])
                label_objectness, label_classification, label_location, label_scale = get_objectness_label(imgs,
                                                                                                           gt_boxes,
                                                                                                           gt_labels,
                                                                                                           anchors=layer_anchor,
                                                                                                           num_classes=num_classes,
                                                                                                           downsample=downsample)
                # print(
                #     f'layer{i}, {label_objectness.shape, label_classification.shape, label_location.shape, label_scale.shape}')
                # layer0, (torch.Size([16, 3, 7, 7]), torch.Size([16, 3, 7, 7, 7]), torch.Size([16, 3, 4, 7, 7]),
                #          torch.Size([16, 3, 7, 7]))
                # layer1, (torch.Size([16, 3, 14, 14]), torch.Size([16, 3, 7, 14, 14]), torch.Size([16, 3, 4, 14, 14]),
                #          torch.Size([16, 3, 14, 14]))
                # layer2, (torch.Size([16, 3, 28, 28]), torch.Size([16, 3, 7, 28, 28]), torch.Size([16, 3, 4, 28, 28]),
                #          torch.Size([16, 3, 28, 28]))
                pred_objectness = layer[:, :, 0, :, :]
                pred_classification = layer[:, :, 1:1+num_classes, :, :]
                pred_location = layer[:, :, 1+num_classes:1+num_classes+4, :, :]
                # print(pred_objectness.shape, pred_classification.shape, pred_location.shape)
                # torch.Size([16, 3, 7, 7])
                # torch.Size([16, 3, 7, 7, 7])
                # torch.Size([16, 3, 4, 7, 7])
                # torch.Size([16, 3, 14, 14])
                # torch.Size([16, 3, 7, 14, 14])
                # torch.Size([16, 3, 4, 14, 14])
                # torch.Size([16, 3, 28, 28])
                # torch.Size([16, 3, 7, 28, 28])
                # torch.Size([16, 3, 4, 28, 28])
                downsample /= 2
            break
        break


if __name__ == '__main__':
    train()
