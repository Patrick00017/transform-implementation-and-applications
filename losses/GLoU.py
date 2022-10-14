import torch


def GLoU(pred_bbox, gt_bbox):
    '''
    compute the generalized intersection over union.
    :param pred_bbox: (x1y1x2y2)
    :param gt_bbox: (x1y1x2y2)
    :return:
    '''
    p_x1, p_y1, p_x2, p_y2 = pred_bbox
    g_x1, g_y1, g_x2, g_y2 = gt_bbox

    area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
    area_g = (g_x2 - g_x1) * (g_y2 - g_y1)

    xmin = max(p_x1, g_x1)
    xmax = min(p_x2, g_x2)
    ymin = max(p_y1, g_y1)
    ymax = min(p_y2, g_y2)

    area_intersection = (xmax - xmin) * (ymax - ymin)
    area_not_intersection = area_p + area_g - area_intersection
    iou = area_intersection / area_not_intersection
    if xmin >= xmax and ymin >= ymax:
        iou = torch.tensor(0).float()
    c_xmin = min(p_x1, g_x1)
    c_xmax = max(p_x2, g_x2)
    c_ymin = min(p_y1, g_y1)
    c_ymax = max(p_y2, g_y2)
    area_c = (c_xmax - c_xmin) * (c_ymax - c_ymin)
    glou = iou - (area_c - area_not_intersection) / area_c
    iou_loss = 1 - iou
    glou_loss = 1 - glou
    return glou_loss


if __name__ == '__main__':
    pred = torch.tensor([0.1, 0.2, 0.2, 0.3])
    gt = torch.tensor([0.15, 0.2, 0.25, 0.4])
    print(GLoU(pred, gt))
