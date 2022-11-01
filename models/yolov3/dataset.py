import numpy as np
import torch
from torch.utils.data import Dataset
import os
import xml.etree.ElementTree as ET
import cv2
from torchvision.transforms.transforms import Compose, ToTensor, Normalize, Resize


def get_annotations(cname2cid, annotations_dir, image_dir):
    filenames = os.listdir(annotations_dir)
    records = []
    ct = 0
    for fname in filenames:
        fid = fname.split('.')[0]
        fpath = os.path.join(annotations_dir, fname)
        img_file = os.path.join(image_dir, fid + '.jpg')
        if fname == '.ipynb_checkpoints':
            continue
        tree = ET.parse(fpath)

        if tree.find('id') is None:
            im_id = np.array([ct])
        else:
            im_id = np.array([int(tree.find('id').text)])

        objs = tree.findall('object')
        im_w = float(tree.find('size').find('width').text)
        im_h = float(tree.find('size').find('height').text)
        gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
        gt_class = np.zeros((len(objs),), dtype=np.int32)
        is_crowd = np.zeros((len(objs),), dtype=np.int32)
        difficult = np.zeros((len(objs),), dtype=np.int32)
        for i, obj in enumerate(objs):
            cname = obj.find('name').text
            gt_class[i] = cname2cid[cname]
            _difficult = int(obj.find('difficult').text)
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)
            # 这里使用xywh格式来表示目标物体真实框
            gt_bbox[i] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1 + 1., y2 - y1 + 1.]
            is_crowd[i] = 0
            difficult[i] = _difficult

        voc_rec = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
        }
        if len(objs) != 0:
            records.append(voc_rec)
        ct += 1
    return records


def get_bbox(gt_bbox, gt_class):
    # 对于一般的检测任务来说，一张图片上往往会有多个目标物体
    # 设置参数MAX_NUM = 50， 即一张图片最多取50个真实框；如果真实
    # 框的数目少于50个，则将不足部分的gt_bbox, gt_class和gt_score的各项数值全设置为0
    MAX_NUM = 50
    gt_bbox2 = np.zeros((MAX_NUM, 4))
    gt_class2 = np.zeros((MAX_NUM,))
    for i in range(len(gt_bbox)):
        gt_bbox2[i, :] = gt_bbox[i, :]
        gt_class2[i] = gt_class[i]
        if i >= MAX_NUM:
            break
    return gt_bbox2, gt_class2


def get_img_data_from_file(record):
    """
    record is a dict as following,
      record = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
            }
    """
    im_file = record['im_file']
    h = record['h']
    w = record['w']
    is_crowd = record['is_crowd']
    gt_class = record['gt_class']
    gt_bbox = record['gt_bbox']
    difficult = record['difficult']

    img = cv2.imread(im_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # check if h and w in record equals that read from img
    assert img.shape[0] == int(h), \
        "image height of {} inconsistent in record({}) and img file({})".format(
            im_file, h, img.shape[0])

    assert img.shape[1] == int(w), \
        "image width of {} inconsistent in record({}) and img file({})".format(
            im_file, w, img.shape[1])

    gt_boxes, gt_labels = get_bbox(gt_bbox, gt_class)

    # gt_bbox 用相对值
    gt_boxes[:, 0] = gt_boxes[:, 0] / float(w)
    gt_boxes[:, 1] = gt_boxes[:, 1] / float(h)
    gt_boxes[:, 2] = gt_boxes[:, 2] / float(w)
    gt_boxes[:, 3] = gt_boxes[:, 3] / float(h)

    return img, gt_boxes, gt_labels, (h, w)


class SilverDataset(Dataset):
    def __init__(self, classname2idx, anno_dir, image_dir, transforms=None, mode='train'):
        self.cname2cid = classname2idx
        self.records = get_annotations(self.cname2cid, anno_dir, image_dir)
        self.img_size = 224  # get_img_size(mode)
        self.transforms = transforms

    def __getitem__(self, idx):
        record = self.records[idx]
        img, gt_bbox, gt_labels, im_shape = get_img_data_from_file(record)
        if self.transforms:
            img = self.transforms(img)
        return img, torch.tensor(gt_bbox), torch.tensor(gt_labels), torch.tensor(im_shape)

    def __len__(self):
        return len(self.records)


def get_train_dataset():
    train_dir_path = 'D:\\code\\python\\datasets\\gangtie_challenge\\train\\train'
    train_image_path = os.path.join(train_dir_path, 'IMAGES')
    train_annotation_path = os.path.join(train_dir_path, 'ANNOTATIONS')
    classname2idx = {'crazing': 0, 'inclusion': 1, 'pitted_surface': 2, 'scratches': 3, 'patches': 4,
                     'rolled-in_scale': 5, 'background': 6}
    idx2classname = [key for key in classname2idx.keys()]
    train_transform = Compose([
        ToTensor(),
        Resize([224, 224]),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = SilverDataset(classname2idx, train_annotation_path, train_image_path, transforms=train_transform)
    return dataset, classname2idx, idx2classname


def get_test_dataset():
    test_image_path = 'D:\\code\\python\\datasets\\gangtie_challenge\\test\\test'
    pass


if __name__ == '__main__':
    get_train_dataset()
