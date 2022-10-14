import os.path
import time

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from models.vit import ViT
from torchvision.transforms import transforms


def train(datasets, epoch_num, optimizer, net, batch_size, criterion, weight_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network = net.to(device)
    loss_function = criterion.to(device)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = None
    if datasets == 'cifar-10':
        dataset = torchvision.datasets.CIFAR10('./datasets', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
    print('start training...')
    total_time_cost = 0
    for epoch in range(1, epoch_num + 1):
        start = time.time()
        sample_num = 0
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            yhat = network(inputs)
            loss = loss_function(yhat, labels)
            loss.backward()
            optimizer.step()

            sample_num += 1
            total_loss += loss.item()
        end = time.time()
        total_time_cost += end - start
        print(f'epoch: {epoch}, mean loss: {total_loss / sample_num}, time: {end - start} seconds')
    print(f'training over. total time cost: {total_time_cost}')
    torch.save(network.state_dict(), weight_path)
    print('saving model weight successfully.')


if __name__ == '__main__':
    image_size = (None, None)
    weight_path = './weights/vit-cifar-10.pth'
    datasets = 'cifar-10'
    if datasets == 'cifar-10':
        image_size = (32, 32)
    net = ViT(image_size=image_size[0], patch_size=4, num_classes=10, dim=128, depth=3, heads=64, mlp_dim=256)
    criterion = nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-5)
    train(datasets='cifar-10', epoch_num=30, optimizer=optimizer, net=net, batch_size=128, criterion=criterion,
          weight_path=weight_path)
