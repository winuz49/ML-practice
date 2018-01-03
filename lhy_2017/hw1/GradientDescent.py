import numpy as np

#TODO
class GradientDescent(object):
    def __init__(self, wt, b, x, y, lr):
        self.__wt = wt
        self.__b = b
        self.__x = x
        self.__y = y
        self.__lr = lr

    @property
    def wt(self):
        return self.__wt

    @property
    def b(self):
        return self.__b

    def train_by_pseudo_inverse(self, x, y, alpha=0, validate_data=None):
        return

    def train(self,x,y,init_wt=np.array([]),init_b=0,rate=0.01,alpha=0,epoch=1000,batch=None,
                validate_data=None):
        return

    def step_gradient(self):
        N = self.__x.shape()[0]
        colums = self.__x.shape()[1]
        w_gradient = np.zeros(colums)
        b_gradient = 0

        for i in range(0, N):
            x = self.__x[i]
            y = self.__y[i]
            w_gradient += -2 * (y - np.dot(w_gradient, x)+b_gradient)*x
            b_gradient += -2 * (y - np.dot(w_gradient, x)+b_gradient)

        new_wt = self.__wt - self.__lr * w_gradient / N
        new_b = self.__b - self.__lr * b_gradient / N

        return [new_wt, new_b]

    def update(self, X, y, wt, b, rate, alpha):
        return

    def predict(self, X):
        return

    def compute_err(self, X, y):
        return

    def _check_data(self, X, y):
        return