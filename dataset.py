import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import xml.etree.ElementTree as ET  # 用来解析.xml文件

#  存储Voc数据集中的类别标签的字典 没打全
voc_class_idx2class = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"]

voc_class_class2idx = dict([(name, idx) for idx, name in enumerate(voc_class_idx2class)])


class VOC2007(Dataset):
    def __init__(self, root_path, transform=None, resize_w=500, resize_h=500):
        super(VOC2007).__init__()
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.transform = transform
        self.root_path = root_path
        self.img_idx = []
        self.anno_idx = []
        self.bbox = []
        self.obj_name = []
        train_txt_path = self.root_path + "/ImageSets/Layout/train.txt"
        self.img_path = self.root_path + "/JPEGImages/"
        self.anno_path = self.root_path + "/Annotations/"
        train_txt = open(train_txt_path)
        lines = train_txt.readlines()
        for l in lines:
            name = l.strip().split()[0]
            self.img_idx.append(self.img_path + name + '.jpg')
            self.anno_idx.append(self.anno_path + name + '.xml')

    def __getitem__(self, item):
        img = torchvision.io.read_image(self.img_idx[item]) / 255.0
        # [‘C’, ‘H’, ‘W’]
        channels, height, width = img.shape
        objects = ET.parse(self.anno_idx[item])
        boxes = []
        clses = []
        for obj in objects.iter('object'):
            cls_and_bbox = []  # [cls_idx, x1, y1, x2, y2]
            name = obj.find('name').text.lower().strip()
            class_idx = voc_class_class2idx[name]
            cls_and_bbox.append(class_idx)
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            box = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                box.append(cur_pt)

            # normalize
            box_normalize = np.array(box) / [width, height, width, height]
            box_normalize = box_normalize.tolist()

            # not normalize
            # box_normalize = np.array(box)

            cls_and_bbox.append(box_normalize[0])
            cls_and_bbox.append(box_normalize[1])
            cls_and_bbox.append(box_normalize[2])
            cls_and_bbox.append(box_normalize[3])
            clses.append(cls_and_bbox[0])
            boxes.extend(cls_and_bbox[1:5])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        clses = torch.tensor(clses, dtype=torch.int64)
        target = {"boxes": boxes, "labels": clses}
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target, self.resize_w, self.resize_h

    def __len__(self):
        data_length = len(self.img_idx)
        return data_length

#  开始调用 Read_data类读取数据，并使用Dataloader生成迭代数据为送入模型中做准备
# Voc_data_path = './VOCdevkit'
# train_data = VOC2007(root_path=Voc_data_path)
# train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

#  可以这样理解：自己定义的Read_data负读取数据，而DataLoader负责按照定义的batch_size指派Read_data去读取
# 指定数目的数据，然后再进行相应的拼接等其它内部操作。
