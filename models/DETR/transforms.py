import random

import numpy as np
import torch
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"].unsqueeze(0)
            bbox[:, [0, 2]] = 1.0 - bbox[:, [2, 0]]
            target["boxes"] = bbox[0]
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class Normalize(object):
    def __init__(self, mean: list, std: list):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        return F.normalize(image, self.mean, self.std), target


class Resize(object):
    def __init__(self, size: list):
        self.size = size

    def __call__(self, image, target):
        return F.resize(image, size=self.size), target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        img_height, img_width = image.shape[-2:]
        targets = []
        for element in target:
            temp = []
            temp.append(element['category_id'])
            x1, y1, w, h = element['bbox']  # x1 y1 w h
            bbox = [x1 / img_width, y1 / img_height, (x1 + w) / img_width, (y1 + h) / img_height]
            temp.extend(bbox)
            targets.append(temp)
        return image, torch.tensor(targets)
