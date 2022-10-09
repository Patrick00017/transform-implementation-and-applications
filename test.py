import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
from models.vit import ViT

classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = npimg[0]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test(image, label, net):
    pred = net(image)
    print(pred.shape)
    imshow(image)
    class_idx = torch.max(pred, 1).indices.item()
    # print(class_idx.indices.item())
    class_name = classes[class_idx]
    print(f'ground truth: {classes[label[0]]}')
    print(f'pred: {class_name}')



if __name__ == '__main__':
    image_size = (32, 32)
    weight_path = './weights/vit-cifar-10.pth'
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    net = ViT(image_size=image_size[0], patch_size=4, num_classes=10, dim=6, depth=2, heads=8, mlp_dim=8)
    net.load_state_dict(torch.load(weight_path))
    net.eval()
    datasets = torchvision.datasets.CIFAR10('./datasets', train=True, transform=transform, download=True)
    train_loader = DataLoader(datasets, batch_size=1, shuffle=True)
    for i in range(10):
        image, label = next(iter(train_loader))
        test(image, label, net)
