import torch
import torch.nn as nn


def get_yolo_loss(pred_objectness, label_objectness, pred_classification, label_classification,
                  pred_location, label_location, scales):
    # print(pred_objectness.shape, label_objectness.shape, pred_classification.shape, label_classification.shape,
    #               pred_location.shape, label_location.shape)
    # torch.Size([16, 3, 7, 7])
    # torch.Size([16, 3, 7, 7])
    # torch.Size([16, 3, 7, 7, 7])
    # torch.Size([16, 3, 7, 7, 7])
    # torch.Size([16, 3, 4, 7, 7])
    # torch.Size([16, 3, 4, 7, 7])
    # torch.Size([16, 3, 7, 7]))
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    softmax_loss = nn.CrossEntropyLoss(reduction='none')
    l1_loss = nn.L1Loss(reduction='none')

    # calculate location loss
    px = pred_location[:, :, 0, :, :]
    py = pred_location[:, :, 1, :, :]
    pw = pred_location[:, :, 2, :, :]
    ph = pred_location[:, :, 3, :, :]
    dx = label_location[:, :, 0, :, :]
    dy = label_location[:, :, 1, :, :]
    dw = label_location[:, :, 2, :, :]
    dh = label_location[:, :, 3, :, :]
    loss_location_x = bce_loss(px, dx)
    loss_location_y = bce_loss(py, dy)
    loss_location_w = l1_loss(pw, dw)
    loss_location_h = l1_loss(ph, dh)
    loss_location = loss_location_x + loss_location_y + loss_location_w + loss_location_h
    loss_location = loss_location * scales

    loss_objectness = bce_loss(pred_objectness, label_objectness)
    loss_cls = softmax_loss(pred_classification, label_classification)
    # print(loss_location.shape, loss_objectness.shape, loss_cls.shape)
    # torch.Size([16, 3, 7, 7])
    # torch.Size([16, 3, 7, 7])
    # torch.Size([16, 7, 7, 7])

    # pos_samples 只有在正样本的地方取值为1.，其它地方取值全为0.
    pos_objectness = label_objectness > 0
    pos_samples = pos_objectness.float()
    pos_samples.requires_grad = False
    # print(pos_samples.shape)
    # torch.Size([16, 3, 7, 7])

    # todo: use this pos sample mask.

    return 0

