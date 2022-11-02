from multiprocessing import Process, Lock, Array
import torch


def f(l, i, arr):
    box = torch.rand(4)
    arr[i, :] = box
    print(f'p{i} finish')


if __name__ == '__main__':
    # loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    # input = torch.tensor([[0.5]]).float()
    # target = torch.tensor([[-1.0]]).float()
    # l = loss(input, target).mean(1)
    # print(input)
    # print(target)
    # print(l)
    # print(l.shape)
    anchors = torch.tensor([[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]])
    print(anchors / 640 * 224)