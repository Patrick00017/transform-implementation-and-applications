import time
import os
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.yolov3.model import Yolov3
from models.yolov3.dataset import get_train_dataset
from tools.bbox import get_objectness_label, convert_txtytwth_2_xyxy, get_iou_above_thresh_inds, label_objectness_ignore
from models.yolov3.loss import get_yolo_loss


batch_size = 16
num_classes = 7  # include playground
num_anchor = 3
epoches = 100
lr = 0.001
wd = 0.0001
momentum = 0.9


def train():
    weight_path = './yolov3_silver_challenge.pth'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_dataset, classname2idx, idx2classname = get_train_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    net = Yolov3(num_classes=num_classes, num_anchor_per_pixel=num_anchor)
    net = net.to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    print('start training...')
    for epoch in range(1, epoches + 1):
        ultimate_loss = torch.tensor(0.0).to(device)
        total_num_images = 0
        epoch_start_time = time.time()
        for i, batch in enumerate(train_loader):
            batch_start_time = time.time()
            imgs, gt_boxes, gt_labels, im_shape = batch
            num_images = imgs.shape[0]
            total_num_images += num_images
            imgs = imgs.to(device)
            optimizer.zero_grad()
            p0, p1, p2 = net(imgs)
            # shape:
            # torch.Size([16, 60, 7, 7]) torch.Size([16, 60, 14, 14]) torch.Size([16, 60, 28, 28])

            # generate multi layer loss and sum together.
            total_loss = torch.tensor(0.0).float().to(device)
            downsample = 32
            anchors = [[40.6000, 31.5000, 54.6000, 69.3000, 130.5500, 114.1000],
                       [10.5000, 21.3500, 21.7000, 15.7500, 20.6500, 41.6500],
                       [3.5000, 4.5500, 5.6000, 10.5000, 11.5500, 8.0500]]
            for j, layer in enumerate([p0, p1, p2]):
                layer_anchor = anchors[j]
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
                label_objectness.requires_grad = False
                label_classification.requires_grad = False
                label_location.requires_grad = False
                label_scale.requires_grad = False
                label_objectness = label_objectness.to('cpu')
                label_classification = label_classification.to('cpu')
                label_location = label_location.to('cpu')
                label_scale = label_scale.to('cpu')
                # print(
                #     f'layer{i}, {label_objectness.shape, label_classification.shape, label_location.shape, label_scale.shape}')
                # layer0, (torch.Size([16, 3, 7, 7]), torch.Size([16, 3, 7, 7, 7]), torch.Size([16, 3, 4, 7, 7]),
                #          torch.Size([16, 3, 7, 7]))
                # layer1, (torch.Size([16, 3, 14, 14]), torch.Size([16, 3, 7, 14, 14]), torch.Size([16, 3, 4, 14, 14]),
                #          torch.Size([16, 3, 14, 14]))
                # layer2, (torch.Size([16, 3, 28, 28]), torch.Size([16, 3, 7, 28, 28]), torch.Size([16, 3, 4, 28, 28]),
                #          torch.Size([16, 3, 28, 28]))
                pred_objectness = layer[:, :, 0, :, :]
                pred_objectness = torch.sigmoid(pred_objectness)
                pred_classification = layer[:, :, 1:1 + num_classes, :, :]
                pred_location = layer[:, :, 1 + num_classes:1 + num_classes + 4, :, :]
                pred_objectness = pred_objectness.to('cpu')
                pred_classification = pred_classification.to('cpu')
                pred_location = pred_location.to('cpu')
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

                # convert the box location from txtytwth to x1y1x2y2
                pred_box = convert_txtytwth_2_xyxy(pred_location, layer_anchor, num_classes, downsample)
                # print(pred_box.shape)
                # torch.Size([16, 7, 7, 3, 4])
                # torch.Size([16, 14, 14, 3, 4])
                # torch.Size([16, 28, 28, 3, 4])

                # select iou threshold above box to keep
                ret_inds = get_iou_above_thresh_inds(pred_box, gt_boxes)
                ret_inds = ret_inds.to('cpu')
                label_objectness = label_objectness_ignore(label_objectness, ret_inds)
                l = get_yolo_loss(pred_objectness, label_objectness, pred_classification, label_classification,
                                  pred_location, label_location, label_scale)
                l = l.to(device)
                total_loss += l
                downsample /= 2
            total_loss.backward()
            optimizer.step()
            batch_end_time = time.time()
            ultimate_loss += total_loss
            if i == 1:
                print(f'batch{i}, loss: {total_loss.item()}, time cost: {batch_end_time - batch_start_time} seconds')
            # print(f'batch{i}, loss: {total_loss.item()}, time cost: {batch_end_time - batch_start_time} seconds')
        epoch_end_time = time.time()
        print(
            f'epoch: {epoch}, mean loss: {(ultimate_loss / total_num_images).item()}, '
            f'time cost: {epoch_end_time - epoch_start_time} seconds')
        if epoch % 5 == 0:
            torch.save(net.state_dict(), weight_path)
            print('save weights successfully.')
    print('train over.')


if __name__ == '__main__':
    train()