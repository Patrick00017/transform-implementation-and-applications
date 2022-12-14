import torch
import torch.nn.functional as F
from losses.GLoU import GLoU
from tools.bbox import xywh_2_xyxy, xyxy_2_xywh


def bounding_box_loss(pred_bbox, gtbox, lou_superparams, l1_superparams):
    """
    box loss function for DETR
    :param pred_bbox: (4)
    :param gtbox: (4)
    :return: loss : tensor(float)
    """
    glou = GLoU(pred_bbox, gtbox)
    p_box = pred_bbox
    gtbox = gtbox
    l1_loss = F.l1_loss(p_box, gtbox)
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
    aim_class = gt_cls.int().item()
    probability = F.softmax(pred_cls, dim=-1)
    aim_class_probability = probability[aim_class]
    class_loss = 1 - aim_class_probability

    box_loss = bounding_box_loss(pred_bbox, gt_box, 1.5, 1)
    return class_loss + box_loss


def hungarian_loss(pred_cls, pred_bbox, gt_clses, gt_boxes, masks, lou_superparams=1.5, l1_superparams=1):
    """
    compute the hungarian loss, straight calculate the cls loss, and use maks to select to calculate the bbox loss.
    :param pred_cls: (B, q, 21)
    :param pred_bbox: (B, q, 4)
    :param gt_clses: (B, q, 1)
    :param gt_boxes: (B, q, 4)
    :param masks: (B, q)
    :param lou_superparams: float
    :param l1_superparams: float
    q = object queries
    :return: loss
    """

    batch_size = pred_cls.shape[0]
    cls_criterion = torch.nn.CrossEntropyLoss()
    cls_loss = cls_criterion(pred_cls.permute(0, 2, 1), gt_clses.squeeze(-1).long())

    bbox_loss = torch.tensor(0).float()
    for b in range(batch_size):
        batch_bbox_loss = torch.tensor(0).float()
        for i in range(masks.shape[1]):
            if masks[b, i] == 1:
                pred_b = pred_bbox[b, i, :]
                pred_b_xyxy = xywh_2_xyxy(pred_b)
                gt_b = gt_boxes[b, i, :]
                bbox_loss = bounding_box_loss(pred_bbox=pred_b_xyxy, gtbox=gt_b, lou_superparams=lou_superparams,
                                              l1_superparams=l1_superparams)
                batch_bbox_loss += bbox_loss
        bbox_loss += batch_bbox_loss

    return cls_loss + bbox_loss / batch_size
