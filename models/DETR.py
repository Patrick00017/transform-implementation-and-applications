import os
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from dataset import VOC2007
from losses.Hungarian import hungarian_loss, match_loss

weight_path = '../weights/detr-voc2007.pth'


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
        pred_bbox = F.sigmoid(self.linear_bbox(h))
        # output shape: torch.Size([1, 100, 21]) torch.Size([1, 100, 4])
        return {'pred_class': pred_class, 'pred_bbox': pred_bbox}


def xavior_init(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)

def generate_labels(pred_class, pred_bbox, gt_boxes):
    '''
    this function is used to generate ground truth pair with the prediction.
    :param pred_class:
    :param pred_bbox:
    :param gt_boxes:
    :return:
    '''
    batch_size, object_queries, _ = pred_class.shape
    # print(f'batch_size: {batch_size}, object queries: {object_queries}')
    num_gt_box = gt_boxes.shape[0]
    gt_cls = gt_boxes[:, 0]
    gt_box = gt_boxes[:, 1:5]
    # (B, queries, cls or x1y1x2y2)
    gt_cls_target = torch.zeros((batch_size, object_queries, 1))
    gt_box_target = torch.zeros_like(pred_bbox)

    # give every gt box best pair.
    allocated_index = torch.zeros((object_queries, 1))
    for i in range(num_gt_box):
        max_match_loss = 0
        best_match_index = -1
        gt = gt_boxes[i]
        for j in range(object_queries):
            # if this object query is allocated, then pass
            if allocated_index[j] == 1:
                continue
            pred_c = pred_class[0, j]
            pred_b = pred_bbox[0, j]
            match_l = match_loss(pred_cls=pred_c, pred_bbox=pred_b, gt_cls=gt[0], gt_box=gt[1:5])
            if match_l > max_match_loss:
                max_match_loss = match_l
                best_match_index = j
        # find the max loss, then set targets for it.
        gt_cls_target[0, best_match_index, 0] = gt[0]
        gt_box_target[0, best_match_index, :] = gt[1:5]
        allocated_index[best_match_index] = 1
    return gt_cls_target, gt_box_target, allocated_index


def train_voc(batch_size=1, epoches=3, learning_rate=0.001, weight_decay=1e-5):
    # init device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # prepare the datasets
    dataset_root_path = 'D:\\code\\python\\datasets\\VOCdevkit\\VOC2007'
    train_data = VOC2007(root_path=dataset_root_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    # network
    net = DETR(num_classes=20)
    net = net.to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
    else:
        net.apply(xavior_init)

    # optimizer
    optimizer = SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
            height, width = batch[0][2], batch[0][3]
            image = image.unsqueeze(0)
            image = image.to(device)
            gt_boxes = torch.tensor(batch[0][1])
            gt_boxes = gt_boxes.to(device)

            optimizer.zero_grad()
            output = net(image)
            pred_class, pred_bbox = output['pred_class'], output['pred_bbox']
            # until this point
            # gtboxes: (gt_box_number, clsx1y1x2y2)
            # pred_class: (batch_size, objectquery, 21(class number))
            # pred_bbox: (batch_size, objectquery, x1y1x2y2)

            # gt_class and gt_bbox shape is like pred_class and pred_bbox, and mask=1 is where gtbox locate
            gt_class, gt_bbox, mask = generate_labels(pred_class, pred_bbox, gt_boxes)
            l = criterian(pred_cls=pred_class, pred_bbox=pred_bbox, gt_cls=gt_class, gt_box=gt_bbox, mask=mask,
                          lou_superparams=1.5, l1_superparams=1)
            l.backward()
            optimizer.step()
            print(f'batch loss: {l.item()}')
            total_loss += l.item()
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'epoch: {epoch}, mean loss: {total_loss / image_num}, time: {epoch_time} seconds')
    print('train over.')
    torch.save(net.state_dict(), weight_path)
    print('save weights successfully.')


def train_coco(batch_size=1, epoches=3, learning_rate=0.01, weight_decay=1e-5):
    # init device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # prepare the datasets
    dataset_root_path = 'D:\\code\\python\\datasets\\VOCdevkit\\VOC2007'

    coco_root_dir = 'root'
    coco_anno_file = 'anno'
    coco_dataset = torchvision.datasets.CocoDetection(root=coco_root_dir, annFile=coco_anno_file)

    # network
    net = DETR(num_classes=20)
    net = net.to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))

    # optimizer
    optimizer = SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # loss function use hungarian loss to criterion
    criterian = hungarian_loss
    print('start training...')
    for epoch in range(1, epoches + 1):
        start_time = time.time()
        image_num = 0
        total_loss = 0
        for i, batch in enumerate(coco_dataset):
            image_num += 1
            image, info = batch
            _, _, img_height, img_width = image.shape
            img = torchvision.transforms.Resize([500, 500])(image)
            # resized image
            img = img.to(device)
            gt_boxes = []
            for idx, annotation in enumerate(info):
                # bbox为检测框的位置坐标
                x_min, y_min, width, height = annotation['bbox']
                x_max, y_max = x_min + width, y_min + height
                x_min /= img_width
                x_max /= img_width
                y_min /= img_height
                y_max /= img_height
                # read class annotation.
                cls = annotation['class']
                gt_box = torch.tensor([cls, x_min, y_min, x_max, y_max])
                gt_boxes.append(gt_box)
            gt_boxes = torch.tensor(gt_boxes)
            gt_boxes = gt_boxes.to(device)

            optimizer.zero_grad()
            output = net(img)
            pred_class, pred_bbox = output['pred_class'], output['pred_bbox']
            # until this point
            # gtboxes: (gt_box_number, clsx1y1x2y2)
            # pred_class: (batch_size, objectquery, 21(class number))
            # pred_bbox: (batch_size, objectquery, x1y1x2y2)

            # gt_class and gt_bbox shape is like pred_class and pred_bbox, and mask=1 is where gtbox locate
            gt_class, gt_bbox, mask = generate_labels(pred_class, pred_bbox, gt_boxes)
            l = criterian(pred_cls=pred_class, pred_bbox=pred_bbox, gt_cls=gt_class, gt_box=gt_bbox, mask=mask,
                          lou_superparams=1.5, l1_superparams=1)
            l.backward()
            optimizer.step()
            total_loss += l.item()
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'epoch: {epoch}, mean loss: {total_loss / image_num}, time: {epoch_time} seconds')
    print('train over.')
    torch.save(net.state_dict(), weight_path)
    print('save weights successfully.')


if __name__ == '__main__':
    # net = DETR(num_classes=20)
    # print(net()['pred_class'].shape, net()['pred_bbox'].shape)
    train_voc()
