import torch
import torch.nn as nn
from models.darknet_53.model import conv_batch, darknet53


class YoloDetectionBlock(nn.Module):
    # define YOLOv3 detection head
    # 使用多层卷积和BN提取特征
    def __init__(self, ch_in, ch_out, is_test=True):
        super(YoloDetectionBlock, self).__init__()

        assert ch_out % 2 == 0, \
            "channel {} cannot be divided by 2".format(ch_out)

        self.conv0 = conv_batch(
            in_num=ch_in,
            out_num=ch_out,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv1 = conv_batch(
            in_num=ch_out,
            out_num=ch_out * 2,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv2 = conv_batch(
            in_num=ch_out * 2,
            out_num=ch_out,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv3 = conv_batch(
            in_num=ch_out,
            out_num=ch_out * 2,
            kernel_size=3,
            stride=1,
            padding=1)
        self.route = conv_batch(
            in_num=ch_out * 2,
            out_num=ch_out,
            kernel_size=1,
            stride=1,
            padding=0)
        self.tip = conv_batch(
            in_num=ch_out,
            out_num=ch_out,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return tip


def upsample(in_num, out_num):
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        conv_batch(in_num, out_num)
    )


class Yolov3(nn.Module):
    def __init__(self, num_classes, num_anchor_per_pixel):
        super(Yolov3, self).__init__()
        self.num_classes = num_classes
        self.num_anchor_per_pixel = num_anchor_per_pixel
        # c0, c1, c2 torch.Size([1, 1024, 8, 8]) torch.Size([1, 512, 16, 16]) torch.Size([1, 256, 32, 32])
        self.backbone = darknet53(num_classes=self.num_classes)
        self.backbone_c0_channel = 1024
        self.backbone_c1_channel = 512
        self.backbone_c2_channel = 256
        self.out_channel = self.num_anchor_per_pixel * (self.num_classes + 5)
        self.upsample_c0_2_c1 = upsample(self.backbone_c0_channel, self.backbone_c1_channel)
        self.upsample_c1_2_c2 = upsample(self.backbone_c1_channel, self.backbone_c2_channel)
        self.head_c0 = YoloDetectionBlock(self.backbone_c0_channel, self.out_channel)
        self.head_c1 = YoloDetectionBlock(self.backbone_c1_channel, self.out_channel)
        self.head_c2 = YoloDetectionBlock(self.backbone_c2_channel, self.out_channel)

    def forward(self, x):
        _, c0, c1, c2 = self.backbone(x)
        p0 = self.head_c0(c0)
        c0_2_c1 = self.upsample_c0_2_c1(c0)
        p1 = self.head_c1(c0_2_c1 + c1)
        c1_2_c2 = self.upsample_c1_2_c2(c1)
        p2 = self.head_c2(c1_2_c2 + c2)
        return p0, p1, p2


if __name__ == '__main__':
    net = Yolov3(7, 3)
    img = torch.rand((1, 3, 224, 224))
    p0, p1, p2 = net(img)
    print(p0.shape, p1.shape, p2.shape)
