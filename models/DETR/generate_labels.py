import torch
from losses.Hungarian import match_loss
from tools.bbox import xyxy_2_xywh, xywh_2_xyxy


def generate_labels(pred_class, pred_bbox, annotations):
    '''
    this function is used to generate ground truth pair with the prediction.
    :param pred_class: (B, 100, 21)
    :param pred_bbox: (B, 100, 4 -> x1y1x2y2)
    :param gt_boxes: list( (gt_num, 5 -> class x1 y1 x2 y2) )
    :return:
    '''

    batch_size, object_queries, _ = pred_class.shape
    # (B, queries, cls or x1y1x2y2)
    gt_cls_target = torch.zeros((batch_size, object_queries, 1))
    gt_box_target = torch.zeros_like(pred_bbox)
    mask = []
    for i, gt_boxes in enumerate(annotations):
        # give every gt box best pair.
        allocated_index = torch.zeros((object_queries, 1))
        for j in range(gt_boxes.shape[0]):
            max_match_loss = 0
            best_match_index = -1
            gt = gt_boxes[j]
            for k in range(object_queries):
                # if this object query is allocated, then pass
                if allocated_index[k] == 1:
                    continue
                pred_c = pred_class[i, k]
                pred_b = pred_bbox[i, k]
                pred_b_xyxy = xywh_2_xyxy(pred_b)
                match_l = match_loss(pred_cls=pred_c, pred_bbox=pred_b_xyxy, gt_cls=gt[0], gt_box=gt[1:5])
                if match_l < max_match_loss:
                    max_match_loss = match_l
                    best_match_index = j
            # find the max loss, then set targets for it.
            gt_cls_target[i, best_match_index, 0] = gt[0]
            gt_box_target[i, best_match_index, :] = gt[1:5]
            allocated_index[best_match_index] = 1
        mask.append(allocated_index.unsqueeze(0))

    return gt_cls_target, gt_box_target, torch.cat(mask, dim=0)
