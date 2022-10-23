import os
import time

import torch
import torchvision
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import VOC2007
from losses.Hungarian import hungarian_loss
from DETR import DETR
from generate_labels import generate_labels
from transforms import Compose, RandomHorizontalFlip, Resize, Normalize, ToTensor

weight_path = '../../weights/detr-voc2007-small.pth'


def xavior_init(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)


def train_voc(batch_size=1, epoches=3, learning_rate=0.01, weight_decay=1e-4, located="425"):
    # init device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # prepare the datasets
    dataset_root_path = '../../datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
    if located == '1414':
        dataset_root_path = 'D:\\code\\python\\datasets\\VOCdevkit\\VOC2007'

    # set transforms
    transform = Compose(
        [
            RandomHorizontalFlip(0.5),
            Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            Resize([500, 500])
        ]
    )
    train_data = VOC2007(root_path=dataset_root_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    # network
    net = DETR(num_classes=20)
    net = net.to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
    else:
        net.apply(xavior_init)

    # optimizer
    optimizer = SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    # loss function use hungarian loss to criterion
    criterian = hungarian_loss
    print('start training...')
    for epoch in range(1, epoches + 1):
        start_time = time.time()
        image_num = 0
        total_loss = 0
        for i, batch in enumerate(train_loader):
            image_num += 1
            image = batch[0][0]
            width, height = batch[0][2], batch[0][3]
            image = image.unsqueeze(0)
            image = image.to(device)
            targets = batch[0][1]
            gt_clses = targets["labels"].unsqueeze(1).to('cpu')
            gt_bboxes = targets["boxes"].reshape(-1, 4).to('cpu')
            gt_boxes = torch.cat((gt_clses, gt_bboxes), dim=1).to('cpu')

            optimizer.zero_grad()
            output = net(image)
            pred_class, pred_bbox = output['pred_class'], output['pred_bbox']
            pred_class = pred_class.to('cpu')
            pred_bbox = pred_bbox.to('cpu')
            # until this point
            # gtboxes: (gt_box_number, clsx1y1x2y2)
            # pred_class: (batch_size, objectquery, 21(class number))
            # pred_bbox: (batch_size, objectquery, x1y1x2y2)

            # gt_class and gt_bbox shape is like pred_class and pred_bbox, and mask=1 is where gtbox locate
            # generate_start_time = time.time()
            gt_class, gt_bbox, mask = generate_labels(pred_class, pred_bbox, gt_boxes)
            # generate_end_time = time.time()
            l = criterian(pred_cls=pred_class, pred_bbox=pred_bbox, gt_cls=gt_class, gt_box=gt_bbox, mask=mask,
                          lou_superparams=1.5, l1_superparams=1)
            l.backward()
            optimizer.step()
            total_loss += abs(l.item())
            # print(f'batch loss: {l.item()}')
            # batch_end_time = time.time()
            # print(f'batch loss: {l.item()}, batch time: {batch_end_time-batch_start_time}s')
            # print(f'generate label time: {generate_end_time-generate_start_time}s')
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'epoch: {epoch}, mean loss: {total_loss / image_num}, time: {epoch_time} seconds')
    print('train over.')
    torch.save(net.state_dict(), weight_path)
    print('save weights successfully.')


def train_coco(batch_size=1, epoches=3, learning_rate=0.001, weight_decay=1e-5):
    weight_path = '../../weights/detr-coco2017.pth'
    # init device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # prepare the datasets
    print('loading datasets: coco2017...')
    # set transforms
    transform = Compose(
        [
            ToTensor(),
            Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            Resize([500, 500])
        ]
    )
    dataset_root_path = 'D:\\datasets\\coco2017'
    train_path = os.path.join(dataset_root_path, 'train2017', 'train2017')
    train_annotation_path = os.path.join(dataset_root_path, 'annotations_trainval2017', 'annotations',
                                         'instances_train2017.json')
    coco_dataset = torchvision.datasets.CocoDetection(root=train_path, annFile=train_annotation_path,
                                                      transforms=transform)
    coco_train_loader = DataLoader(coco_dataset, batch_size=1, shuffle=True)
    print(f'Number of samples: {len(coco_dataset)}.')
    # for batch in coco_train_loader:
    #     img, target = batch
    #     # img = img.unsqueeze(0).to(device)
    #     print(img.shape)
    #     print(target.shape)
    #     break
    # return
    # network
    net = DETR(num_classes=90)
    net = net.to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))

    # optimizer
    optimizer = SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    # loss function use hungarian loss to criterion
    criterian = hungarian_loss
    print('start training...')
    for epoch in range(1, epoches + 1):
        start_time = time.time()
        image_num = 0
        total_loss = 0
        for i, batch in enumerate(coco_train_loader):
            image_num += 1
            image, gt_boxes = batch
            width, height = 500, 500
            image = image.to(device)
            gt_boxes = gt_boxes[0].to(device)

            optimizer.zero_grad()
            output = net(image)
            pred_class, pred_bbox = output['pred_class'], output['pred_bbox']
            pred_class = pred_class.to('cpu')
            pred_bbox = pred_bbox.to('cpu')
            # until this point
            # gtboxes: (gt_box_number, clsx1y1x2y2)
            # pred_class: (batch_size, objectquery, 21(class number))
            # pred_bbox: (batch_size, objectquery, x1y1x2y2)

            # gt_class and gt_bbox shape is like pred_class and pred_bbox, and mask=1 is where gtbox locate
            # generate_start_time = time.time()
            gt_class, gt_bbox, mask = generate_labels(pred_class, pred_bbox, gt_boxes)
            # generate_end_time = time.time()
            l = criterian(pred_cls=pred_class, pred_bbox=pred_bbox, gt_cls=gt_class, gt_box=gt_bbox, mask=mask,
                          lou_superparams=1.5, l1_superparams=1)
            l.backward()
            optimizer.step()
            total_loss += abs(l.item())
            if image_num >= 100:
                break
            # print(f'batch loss: {l.item()}')
            # batch_end_time = time.time()
            # print(f'batch loss: {l.item()}, batch time: {batch_end_time-batch_start_time}s')
            # print(f'generate label time: {generate_end_time-generate_start_time}s')
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'epoch: {epoch}, mean loss: {total_loss / image_num}, time: {epoch_time} seconds')
    print('train over.')
    torch.save(net.state_dict(), weight_path)
    print('save weights successfully.')


if __name__ == '__main__':
    # net = DETR(num_classes=20)
    # print(net()['pred_class'].shape, net()['pred_bbox'].shape)
    # train_voc(epoches=50, learning_rate=0.001, located='1414')
    train_coco(epoches=10)
