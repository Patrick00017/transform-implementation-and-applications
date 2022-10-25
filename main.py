from multiprocessing import Process, Lock, Array
import torch


def f(l, i, arr):
    box = torch.rand(4)
    arr[i, :] = box
    print(f'p{i} finish')

if __name__ == '__main__':
    lock = Lock()
    shape = (3, 4)
    arr = torch.zeros(size=shape).share_memory_()
    print(f'before: {arr}')
    process = []
    for num in range(3):
        a = Process(target=f, args=(lock, num, arr))
        a.start()
        process.append(a)
    for p in process:
        p.join()

    print(f'after: {arr}')
