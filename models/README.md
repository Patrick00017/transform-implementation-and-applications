# 图像识别
## 网络结构
### Resnet50 结构图
![Resnet50](./images/The-architecture-of-ResNet-50.png)

### ViT Transformer 结构图
![ViT Transfomer](./images/vision-transformer-vit.webp)

## 实验和实验结果
### 数据集
CIFAR-10 batchsize=128

### 网络
vit/tiny:  patchsize:4 dim=128 depth=3, heads=64, mlp_dim=256 params=12798858

vit/base:  patchsize:4 dim=128 depth=3, heads=64, mlp_dim=256 params=21320586

resnet50:                                                     params=23528522



### 实验
优化器：sgd(learning_rate=1e-3, weight_decay=1e-4, momentum=0.9)

在图像增广前，ViT/tiny训练200轮, Resnet50使用预训练权重在CIFAR-10上微调100轮：

| 模型名称 | 参数     | 准确率 | 时间花费             |
|----------|----------|--------|----------------------|
| vit/tiny | 12798858 | 0.603  | 35.405102252960205秒 |
| vit/base | 21320586 | ...    | ...                  |
| resnet50 | 23528522 | 0.68   | 26.82456350326538秒  |

加入图像增广随机裁剪和随机水平翻转后，模型依然使用相同优化器微调100轮，结果：

| 模型名称 | 参数     | 准确率 | 时间花费                |
|----------|----------|--------|---------------------|
| vit/tiny | 12798858 | 0.7793  | 35.405102252960205秒 |
| vit/base | 21320586 | 0.7938    | 35.0678288936615秒   |
| resnet50 | 23528522 | 0.8348   | 26.82456350326538秒  |

