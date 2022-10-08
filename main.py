import time

if __name__ == '__main__':
    start = time.time()
    for i in range(100000):
        print('PyCharm')
    end = time.time()
    print(end - start)


