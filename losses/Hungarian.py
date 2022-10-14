import torch
import torch.nn.functional as F
from GLoU import GLoU


def bounding_box_loss(pred_bbox, gtbox, lou_superparams, l1_superparams):
    """
    box loss function for DETR
    :param pred_bbox: (4)
    :param gtbox: (4)
    :return: loss : tensor(float)
    """
    glou = GLoU(pred_bbox, gtbox)
    l1_loss = F.l1_loss(pred_bbox, gtbox)
    return lou_superparams * glou + l1_superparams * l1_loss


def match_loss(pred_cls, pred_bbox, gt_cls, gt_box):
    """
    this function is used to define how nearest between the pred and the gt by also class and box.
    if the pred and the gt want to be a pair, this match loss should be the max loss comparing to the other prediction.
    :param pred_cls: (21)
    :param pred_bbox: (4)
    :param gt_cls: (1)
    :param gt_box: (4)
    :return:
    """
    aim_class = gt_cls[0]
    probability = F.softmax(pred_cls, dim=-1)
    aim_class_probability = probability[aim_class]
    class_loss = 1 - aim_class_probability

    box_loss = bounding_box_loss(pred_bbox, gt_box, 1, 1)
    return -class_loss + box_loss


def hungarian_loss(preds, gtboxes, lou_superparams=1, l1_superparams=1):
    pass
