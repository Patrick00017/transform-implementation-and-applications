import torch
from losses.Hungarian import match_loss
from tools.bbox import xyxy_2_xywh, xywh_2_xyxy
from multiprocessing import Value, Lock, Process


def generate(lock, b, batch_size, object_queries, annotations, pred_class, pred_bbox, gt_cls_target, gt_box_target,
             mask):
    while True:
        lock.acquire()
        if b.value >= batch_size:
            lock.release()
            break
        B = b.value
        b.value = b.value + 1
        lock.release()
        # give every gt box best pair.
        allocated_index = torch.zeros((object_queries,), requires_grad=False)
        gt_boxes = annotations[B]
        for j in range(gt_boxes.shape[0]):
            max_match_loss = 100
            best_match_index = -1
            gt = gt_boxes[j]
            for k in range(object_queries):
                # if this object query is allocated, then pass
                if allocated_index[k] == 1:
                    continue
                pred_c = pred_class[B, k]
                pred_b = pred_bbox[B, k]
                pred_b_xyxy = xywh_2_xyxy(pred_b)
                match_l = match_loss(pred_cls=pred_c, pred_bbox=pred_b_xyxy, gt_cls=gt[0], gt_box=gt[1:5])
                if match_l.item() < max_match_loss:
                    max_match_loss = match_l.item()
                    best_match_index = k
            # find the max loss, then set targets for it.
            gt_cls_target[B, best_match_index, 0] = gt[0]
            gt_box_target[B, best_match_index, :] = gt[1:5]
            allocated_index[best_match_index] = 1
        mask[B, :] = allocated_index


def generate_labels(pred_class, pred_bbox, annotations, process_num):
    '''
    this function is used to generate ground truth pair with the prediction.
    :param pred_class: (B, 100, 21)
    :param pred_bbox: (B, 100, 4 -> x1y1x2y2)
    :param annotations: list( (gt_num, 5 -> class x1 y1 x2 y2) )
    :return:
    '''

    batch_size, object_queries, _ = pred_class.shape
    # (B, queries, cls or x1y1x2y2)
    gt_cls_target = torch.zeros((batch_size, object_queries, 1), requires_grad=False).share_memory_()
    gt_box_target = torch.zeros_like(pred_bbox, requires_grad=False).share_memory_()
    mask = torch.zeros(size=(batch_size, object_queries), requires_grad=False).share_memory_()
    pred_class_c = pred_class.clone().detach()
    pred_bbox_c = pred_bbox.clone().detach()
    # for i in range(batch_size):
    #     allocated_index = torch.zeros((object_queries,))
    #     gt_boxes = annotations[i]
    #     for j in range(gt_boxes.shape[0]):
    #         max_match_loss = 100
    #         best_match_index = -1
    #         gt = gt_boxes[j]
    #         for k in range(object_queries):
    #             # if this object query is allocated, then pass
    #             if allocated_index[k] == 1:
    #                 continue
    #             pred_c = pred_class[i, k]
    #             pred_b = pred_bbox[i, k]
    #             pred_b_xyxy = xywh_2_xyxy(pred_b)
    #             match_l = match_loss(pred_cls=pred_c, pred_bbox=pred_b_xyxy, gt_cls=gt[0], gt_box=gt[1:5])
    #             if match_l.item() < max_match_loss:
    #                 max_match_loss = match_l.item()
    #                 best_match_index = k
    #         # find the max loss, then set targets for it.
    #         gt_cls_target[i, best_match_index, 0] = gt[0]
    #         gt_box_target[i, best_match_index, :] = gt[1:5]
    #         allocated_index[best_match_index] = 1
    #     mask[i, :] = allocated_index
    #     print(f'gtbox num: {gt_boxes.shape[0]}, mask: {mask[i]}')

    lock = Lock()
    b = Value('i', 0)
    processes = []
    for i in range(process_num):
        # lock, b, batch_size, object_queries, annotations, pred_class, pred_bbox, gt_cls_target, gt_box_target, mask
        p = Process(target=generate,
                    args=(lock, b, batch_size, object_queries, annotations, pred_class_c, pred_bbox_c, gt_cls_target,
                          gt_box_target, mask))
        p.start()
        processes.append(p)
    for process in processes:
        process.join()
    return gt_cls_target, gt_box_target, mask
