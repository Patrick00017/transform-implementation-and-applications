import torch
from losses.Hungarian import match_loss
from tools.bbox import xyxy_2_xywh, xywh_2_xyxy


def generate_labels(pred_class, pred_bbox, gt_boxes):
    '''
    this function is used to generate ground truth pair with the prediction.
    :param pred_class: (B, 100, 21)
    :param pred_bbox: (B, 100, 4 -> x1y1x2y2)
    :param gt_boxes: (gt_num, 5 -> class x1 y1 x2 y2)
    :return:
    '''

    batch_size, object_queries, _ = pred_class.shape
    num_gt_box = gt_boxes.shape[0]
    pred_class_copy = pred_class.to('cpu')
    pred_bbox_copy = pred_bbox.to('cpu')
    # (B, queries, cls or x1y1x2y2)
    gt_cls_target = torch.zeros((batch_size, object_queries, 1)).to('cpu')
    gt_box_target = torch.zeros_like(pred_bbox).to('cpu')

    # give every gt box best pair.
    allocated_index = torch.zeros((object_queries, 1)).to('cpu')
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
            pred_b_xyxy = xywh_2_xyxy(pred_b)
            match_l = match_loss(pred_cls=pred_c, pred_bbox=pred_b_xyxy, gt_cls=gt[0], gt_box=gt[1:5])
            if match_l > max_match_loss:
                max_match_loss = match_l
                best_match_index = j
        # find the max loss, then set targets for it.
        gt_cls_target[0, best_match_index, 0] = gt[0]
        gt_box_target[0, best_match_index, :] = gt[1:5]
        allocated_index[best_match_index] = 1
    return gt_cls_target, gt_box_target, allocated_index
