import numpy as np


class GradientDescent(object):
    def __init__(self):
        self.__wt = None
        self.__b = None
        self.__x = None
        self.__y = None

    @property
    def wt(self):
        return self.__wt

    @property
    def b(self):
        return self.__b

    def train_by_pseudo_inverse(self, X, y, alpha=0, validate_data=None):
        return

    def train(self,X,y,init_wt=np.array([]),init_b=0,rate=0.01,alpha=0,epoch=1000,batch=None,
                validate_data=None):
        return

    def update(self, X, y, wt, b, rate, alpha):
        return

    def predict(self, X):
        return

    def err(self, X, y):
        return

    def _check_data(self, X, y):
        return