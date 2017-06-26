# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import os
import re

# 使得到的似然函数最大(即是每次预测的结果都尽可能的大) 使用梯度上升求解 得到最大的值

def loadData():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineAtr = line.strip().split()
        dataMat.append([1.0,float(lineAtr[0]),float(lineAtr[1])])
        labelMat.append(int(lineAtr[2]))

    return dataMat,labelMat

# 可能有溢出 使用longfloat解决
def sigmoid(inX):
    #print inX
    return longfloat(1.0/(1+exp(-inX)))

# 使用梯度上升方法 得到theta参数
def gradAscent(dataSet,classLabel):

    dataMat = mat(dataSet)
    labelMat = mat(classLabel).transpose()

    m,n = shape(dataMat)
    maxCycles = 300
    alpha = 0.01
    theta = ones((n,1))
    #print shape(theta)

    for i in range(maxCycles):
        h = sigmoid(dataMat*theta)

        error = (labelMat -h)
        theta = theta + alpha*dataMat.transpose()*error

    return theta
# 随机梯度下降 减少每次训练的量 为了解决大数据计算的问题 不过准确 dataMatrix 可能需要加上array进行转换
def stocFradAscent(dataMatrix,classLabel,numIter=150):
    m,n = shape(dataMatrix)
    theta = ones(n)

    for j in range(numIter):


        dataIndex = range(m)
        for i in range(m):
            alpha = 4.0/(i+j+1.0)+0.01
            index = int(random.uniform(0,len(dataIndex)))
            randIndex = dataIndex[index]
            h = sigmoid(sum(dataMatrix[randIndex]*theta))
            error = classLabel[randIndex]-h
            theta = theta +alpha * error* dataMatrix[randIndex]
            del(dataIndex[index])


    return theta



def plotBestFit(weights):

    dataMat,labelMat=loadData()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


def classifyVector(inX,theta):

    prob = sigmoid(sum(inX*theta))

    if prob >=0.5:
        return 1.0

    return 0.0


def colicTest():

    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')

    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        curLine = line.strip().split()
        #print curLine
        lineAtr = []
        for i in range(21):
            lineAtr.append(float(curLine[i]))
        trainingSet.append(lineAtr)

        trainingLabels.append(float(curLine[21]))

    theta = stocFradAscent(array(trainingSet),trainingLabels,500)
    #theta = gradAscent(trainingSet,trainingLabels)

    errorCount = 0
    numTestVec = 0

    for line in frTest.readlines():
        numTestVec += 1
        curLine = line.strip().split()
        #print curLine
        lineAtr = []
        for i in range(21):
            lineAtr.append(float(curLine[i]))
        testLabel = classifyVector(array(lineAtr),theta)
        if int(testLabel) != int(curLine[21]):
            errorCount +=1

    errorRate = errorCount/float(numTestVec)
    print 'the error rate of this test is %f:' % errorRate

    return errorRate

def multiTest():

    numTest = 10
    errorSum = 0.0

    for i in range(numTest):
        errorSum += colicTest()

    print 'after %d iterations the average error rate is : %f' % (numTest,errorSum/float(numTest))





if __name__ == '__main__':
    dataMat,labelMat = loadData()

    #theta = stocFradAscent(array(dataMat),labelMat)
    # getA matrix和array的转换
    #plotBestFit(theta.getA())
    #plotBestFit(theta)

    #print colicTest()
    multiTest()


