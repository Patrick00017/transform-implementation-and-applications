import torch
import torch.nn.functional as F
from losses.GLoU import GLoU

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
    aim_class = gt_cls.int().item()
    probability = F.softmax(pred_cls, dim=-1)
    aim_class_probability = probability[aim_class]
    class_loss = 1 - aim_class_probability

    box_loss = bounding_box_loss(pred_bbox, gt_box, 1, 1)
    return -class_loss + box_loss


def hungarian_loss(pred_cls, pred_bbox, gt_cls, gt_box, mask, lou_superparams=1.5, l1_superparams=1):
    """
    compute the hungarian loss, straight calculate the cls loss, and use maks to select to calculate the bbox loss.
    :param pred_cls: (B, q, 21)
    :param pred_bbox: (B, q, 4)
    :param gt_cls: (B, q, 1)
    :param gt_box: (B, q, 4)
    :param mask: (q)
    :param lou_superparams: float
    :param l1_superparams: float
    q = object queries
    :return: loss
    """

    cls_criterion = torch.nn.CrossEntropyLoss()
    cls_criterion = cls_criterion.to(device)
    gt_cls = gt_cls.long().squeeze(-1)[0].to(device)
    pred_cls = pred_cls[0].to(device)
    # print(f'pred_cls: {pred_cls.shape}, gt_cls: {gt_cls.shape}')
    cls_loss = cls_criterion(pred_cls, gt_cls)

    total_bbox_loss = torch.tensor(0).float().to(device)
    for i in range(mask.shape[0]):
        if mask[i, 0] == 1:
            pred_b = pred_bbox[0, i, :]
            gt_b = gt_box[0, i, :]
            bbox_loss = bounding_box_loss(pred_bbox=pred_b, gtbox=gt_b, lou_superparams=lou_superparams,
                                          l1_superparams=l1_superparams)
            total_bbox_loss += bbox_loss
    return cls_loss + total_bbox_loss
