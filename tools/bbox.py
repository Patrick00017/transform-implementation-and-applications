import torch
import numpy as np


def xyxy_2_xywh(box):
    """
    convert the box formulation.
    :param box: (x1y1x2y2)
    :return: box: (xywh)
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width / 2
    cy = y1 + height / 2
    return torch.tensor([cx, cy, width, height])


def xywh_2_xyxy(box):
    cx, cy, width, height = box
    x1 = cx - width / 2
    x2 = cx + width / 2
    y1 = cy - height / 2
    y2 = cy + height / 2
    return torch.tensor([x1, y1, x2, y2])


# 计算IoU，矩形框的坐标形式为xywh
def box_iou_xywh(box1, box2):
    x1min, y1min = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
    x1max, y1max = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
    s1 = box1[2] * box1[3]

    x2min, y2min = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
    x2max, y2max = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0
    s2 = box2[2] * box2[3]

    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    inter_h = np.maximum(ymax - ymin, 0.)
    inter_w = np.maximum(xmax - xmin, 0.)
    intersection = inter_h * inter_w

    union = s1 + s2 - intersection
    iou = intersection / union
    return iou


def get_objectness_label(img, gt_boxes, gt_labels, iou_threshold=0.7, anchors=None, num_classes=7, downsample=32):
    """
        img 是输入的图像数据，形状是[N, C, H, W]
        gt_boxes，真实框，维度是[N, 50, 4]，其中50是真实框数目的上限，当图片中真实框不足50个时，不足部分的坐标全为0
        真实框坐标格式是xywh，这里使用相对值
        gt_labels，真实框所属类别，维度是[N, 50]
        iou_threshold，当预测框与真实框的iou大于iou_threshold时不将其看作是负样本
        anchors，锚框可选的尺寸
        anchor_masks，通过与anchors一起确定本层级的特征图应该选用多大尺寸的锚框
        num_classes，类别数目
        downsample，特征图相对于输入网络的图片尺寸变化的比例
        """
    if anchors is None:
        anchors = [116, 90, 156, 198, 373, 326]
    img_shape = img.shape
    batch_size = img_shape[0]
    im_height, im_width = img_shape[2:4]
    num_anchors = gt_boxes.shape[1]

    # 计算单元格数量
    num_rows = im_height // downsample
    num_cols = im_width // downsample

    label_objectness = np.zeros((batch_size, num_anchors, num_rows, num_cols))
    label_classification = np.zeros((batch_size, num_anchors, num_classes, num_rows, num_cols))
    label_location = np.zeros((batch_size, num_anchors, 4, num_rows, num_cols))
    scale_location = np.zeros((batch_size, num_anchors, num_rows, num_cols))

    # 逐个图片进行处理
    for n in range(batch_size):
        # 遍历每个真实框
        for n_gt in range(len(gt_boxes[n])):
            gt = gt_boxes[n][n_gt]
            gt_cls = gt_labels[n][n_gt]
            gt_center_x, gt_center_y, gt_width, gt_height = gt
            # 注意：FCOS也可以在这个位置使用stride比对大小跳过标注预测框
            if (gt_width < 1e-3) or (gt_height < 1e-3):
                continue
            i = int(gt_center_y * num_rows)
            j = int(gt_center_x * num_cols)
            ious = []
            for ka in range(num_anchors):
                # 真实框
                box1 = [0.0, 0.0, float(gt_width), float(gt_height)]
                # 选择锚框长和宽
                anchor_w = anchors[ka * 2]
                anchor_h = anchors[ka * 2 + 1]
                box2 = [0.0, 0.0, float(anchor_w), float(anchor_h)]
                iou = box_iou_xywh(box1, box2)
                ious.append(iou)
            ious = np.array(ious)
            inds = np.argsort(ious)
            k = inds[-1]
            label_objectness[n, k, i, j] = 1
            c = gt_cls
            label_classification[n, k, c, i, j] = 1.

            # 为objectness为1的位置设置位置偏移量
            dx_label = gt_center_x * num_cols - j
            dy_label = gt_center_y * num_rows - i
            dw_label = np.log(gt_width * im_width / anchors[k * 2])
            dh_label = np.log(gt_height * im_height / anchors[k * 2 + 1])
            label_location[n, k, 0, i, j] = dx_label
            label_location[n, k, 1, i, j] = dy_label
            label_location[n, k, 2, i, j] = dw_label
            label_location[n, k, 3, i, j] = dh_label
            # scale_location用来调节不同尺寸的锚框对损失函数的贡献，作为加权系数和位置损失函数相乘
            scale_location[n, k, i, j] = 2.0 - gt_width * gt_height
    return torch.from_numpy(label_objectness), \
           torch.from_numpy(label_classification), \
           torch.from_numpy(label_location), \
           torch.from_numpy(scale_location)
