import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# sinx
def real_func(x):
    return np.sin(2*np.pi*x)

# a+bx+cx^2+dx^3+ex^4+...
def fit_func(p, x):
    func = np.poly1d(p)
    return func(x)


def res_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret


def res_func_regularization(p, x, y, regularization=5):
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(0.5*regularization*np.square(p)))
    return ret


def fitting(x, y, func=res_func, m=0):
    print("fitting")
    p_init = np.random.rand(m+1)
    print(p_init)
    p_lsq = leastsq(func, p_init, args=(x, y))
    print("fitting parameters:", p_lsq[0])

    return p_lsq


def display(x_points, p_lsq, x, y):
    print("display")
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    x = np.linspace(0, 1, 10)
    print(x)
    y_ = real_func(x)
    y = [np.random.normal(0, 0.1) + y_tmp for y_tmp in y_]
    p_lsq = fitting(x, y, res_func, 9)
    p_lsq_reg = fitting(x, y, res_func_regularization, 9)
    x_points = np.linspace(0, 1, 1000)
    #display(x_points, p_lsq, x, y)
    display(x_points, p_lsq_reg, x, y)


