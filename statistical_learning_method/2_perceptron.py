import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


class Model:
    def __init__(self, length):
        self.w = np.ones(length, dtype=np.float32)
        self.b = 0
        self.learning_rate = 0.1

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    def fit(self, x_train, y_train):

        need_train = True

        while need_train:
            wrong_count = 0
            for i in range(len(x_train)):
                x = x_train[i]
                y = y_train[i]

                if y * self.sign(x, self.w, self.b) < 0:
                    wrong_count = wrong_count+1
                    self.w = self.w + self.learning_rate * np.dot(y, x)
                    self.b = self.b + self.learning_rate * y
            if wrong_count == 0:
                break

        return


def display(df):
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


def display_result(perceptron):
    x_points = np.linspace(4, 7, 10)
    y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
    plt.plot(x_points, y_)

    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


def sklearn_perceptron(x_train, y_train):
    clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
    clf.fit(x_train, y_train)
    print(clf.coef_)
    print(clf.intercept_)
    x_ponits = np.arange(4, 8)
    y_ = -(clf.coef_[0][0] * x_ponits + clf.intercept_) / clf.coef_[0][1]
    plt.plot(x_ponits, y_)

    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("hello perceptron")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    print(df.label)
    print(df.columns)
    tmp = np.ones(10)
    print(tmp)
    display(df)

    data = np.array(df.iloc[:100, [0, 1, -1]])
    x_train, y_train = data[:, :-1], data[:, -1]
    y_train = np.array([1 if i == 1 else -1 for i in y_train])
    print(x_train)
    print(y_train)

    model = Model(len(x_train[0]))
    model.fit(x_train, y_train)
    print(model.w, model.b)
    display_result(model)
    sklearn_perceptron(x_train, y_train)

