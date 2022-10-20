import torch


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
