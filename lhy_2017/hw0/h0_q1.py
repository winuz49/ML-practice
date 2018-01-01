import numpy as np

a = np.loadtxt('matrixA.txt')
b = np.loadtxt('matrixB.txt')
c = a.dot(b).reshape([1, -1])
c.sort()
count = 1
with open('./ans_one.txt', 'wt') as f:
    for x in np.nditer(c):
        f.write('{} {}\n'.format(count, int(x)))
        count += 1


