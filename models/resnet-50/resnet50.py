import os
import time

import numpy as np
import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.transforms import transforms
import torch.nn.functional as F
from models.net_params_count import count_parameters

weight_path = '../../weights/resnet50-cifar10.pth'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(2048, self.num_classes)
        # self.dense = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x


def finetune(batch_size=1024, epoches=50, learning_rate=0.001):
    net = Resnet50(num_classes=10)
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10('../../datasets', train=True, transform=transform, download=True)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
    print('start training...')
    total_time_cost = 0
    for epoch in range(1, epoches + 1):
        start = time.time()
        sample_num = 0
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            yhat = net(inputs)
            loss = criterion(yhat, labels)
            loss.backward()
            optimizer.step()

            sample_num += inputs.shape[0]
            total_loss += loss.item()
        end = time.time()
        total_time_cost += end - start
        print(f'epoch: {epoch}, mean loss: {total_loss / sample_num}, time: {end - start} seconds')
    print(f'training over. total time cost: {total_time_cost}')
    torch.save(net.state_dict(), weight_path)
    print('saving model weight successfully.')


def evaluate(weight_path):
    image_size = (32, 32)
    net = Resnet50(num_classes=10)
    net = net.to(device)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    net.load_state_dict(torch.load(weight_path))
    net_params_num = count_parameters(net)
    net.eval()

    batch_size = 512
    datasets = torchvision.datasets.CIFAR10('../../datasets', train=False, transform=transform, download=True)
    test_loader = DataLoader(datasets, batch_size=batch_size, shuffle=False, num_workers=2)
    total_right_nums = 0
    total_nums = 0
    y = []
    yhat = []
    with torch.no_grad():
        start_time = time.time()
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            image_num = images.shape[0]
            total_nums += image_num
            preds = net(images)
            preds = F.softmax(preds, dim=-1)
            preds = torch.argmax(preds, dim=-1)
            y.extend(labels.tolist())
            yhat.extend(preds.tolist())
            for i in range(image_num):
                if labels[i] == preds[i]:
                    total_right_nums += 1
        end_time = time.time()
    time_cost = end_time - start_time

    # data = np.zeros((len(yhat), len(y)))
    # for i, a in enumerate(yhat):
    #     for j, b in enumerate(y):
    #         if a == b:
    #             data[i, j] = 1.0
    # print(data)
    # fig, ax = plt.subplots()
    # ax.scatter(yhat, y, s=sizes, c=colors, vmin=0, vmax=100)
    # plt.imshow(data, cmap='gray')
    # plt.show()
    return total_right_nums / total_nums, net_params_num, time_cost


if __name__ == '__main__':
    # finetune()
    precition, params, time_cost = evaluate(weight_path)
    print(precition, params, time_cost)
