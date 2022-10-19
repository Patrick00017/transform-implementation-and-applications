import numpy as np
import torch
import torch.nn as nn
import einops
import torchvision
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


def pair(t):
    """
    make t to be a pair (tuple)
    :param t: object
    :return: a pair(tuple)
    """
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    """
    This block is to be used to normalize the data by layer style
    dim: dimension input
    fn: next net block after the LayerNorm
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    This block is made by the fully connect layer and not a big deal.
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim * dim_head
        # decide to output or not.
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(p=dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # k transpose means: k(b h n d) -> transpose(-1,-2) -> k(b h d n)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = einops.repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def evaluate(weight_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_size = (32, 32)
    net = ViT(image_size=image_size[0], patch_size=4, num_classes=10, dim=128, depth=3, heads=64, mlp_dim=256)
    net = net.to(device)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    net.load_state_dict(torch.load(weight_path))
    net.eval()

    batch_size = 128
    datasets = torchvision.datasets.CIFAR10('../datasets', train=False, transform=transform, download=True)
    test_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=2)
    total_right_nums = 0
    total_nums = 0
    y = []
    yhat = []
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            image_num = images.shape[0]
            total_nums += image_num
            preds = net(images)
            labels = labels.to('cpu')
            preds = preds.to('cpu')
            preds = F.softmax(preds, dim=-1)
            preds = torch.argmax(preds, dim=-1)
            y.extend(labels.tolist())
            yhat.extend(preds.tolist())
            for i in range(image_num):
                if labels[i] == preds[i]:
                    total_right_nums += 1

    data = np.zeros(len(yhat), len(y))
    for i, a in enumerate(yhat):
        for j, b in enumerate(y):
            if a == b:
                data[i, j] = 1
    fig, ax = plt.subplots()
    # ax.scatter(yhat, y, s=sizes, c=colors, vmin=0, vmax=100)
    plt.imshow(data)
    plt.show()
    return total_right_nums / total_nums


if __name__ == '__main__':
    weight_path = '../weights/vit-cifar-10.pth'
    mean_accurate = evaluate(weight_path=weight_path)
    print(mean_accurate)
