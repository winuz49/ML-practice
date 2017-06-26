# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import os
from math import log
import operator
import testPlot
import pickle


# 计算目标变量(向量最后一位)的熵
def calcShannonEnt(dataSet):

    labelCount = {}
    numEntries = len(dataSet)
    for featureVec in dataSet:
        # 需要计算的变量的位置
        currentLabel = featureVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    shanNonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key])/numEntries
        shanNonEnt -=prob * log(prob,2)

    return shanNonEnt

# 测试数据集
def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no'] ]
    label = ['no surfacing','flippers']

    return dataSet,label


#根据特征值划分数据集
def splitDataSet(dataSet,axis,value):
    resetDataSet=[]
    for featureVec in dataSet:
        if featureVec[axis] == value :
            reduceVec=featureVec[:axis]
            reduceVec.extend(featureVec[axis+1:])
            resetDataSet.append(reduceVec)
    return resetDataSet

# 根据特征划分的信息增益 选取最佳的特征
# 公式为g(D,A) = H(D)-H(D|A)

def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0])-1
    baseEntroy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeature):
        featureList = [featureVec[i] for featureVec in dataSet]
        uniqueVal = set(featureList)
        newEntroy = 0.0
        for value in uniqueVal:
            splitData = splitDataSet(dataSet,i,value)
            prob = float(len(splitData)) /float(len(dataSet))
            newEntroy += prob*calcShannonEnt(splitData)
        infoGain = baseEntroy-newEntroy
        #print i,infoGain
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

#输出频度最高的
def majorityCnt(classList):
    classCount = {}

    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount =sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList = [featureVec[-1] for featureVec in dataSet]
    # 划分的类别都一样 返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 已经遍历所有的特征 返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree={bestFeatureLabel:{}}
    del(labels[bestFeature])

    featureValues = [featureVec[bestFeature] for featureVec in dataSet]
    uniqueValue =set(featureValues)

    for value in uniqueValue:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value]=createTree(splitDataSet(dataSet,bestFeature,value),subLabels)


    return myTree

# 使用构建好的决策树进行测试
def classify(inTree , featureLabels,testVec):

    firstStr = inTree.keys()[0]
    secondDict = inTree[firstStr]
    testIndex = featureLabels.index(firstStr)

    for key in secondDict.keys():
        if key == testVec[testIndex]:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featureLabels,testVec)
            else :
                classLabel = secondDict[key]

    return classLabel

#对象序列化
def storeTree(inputTree,filename):
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)



if __name__=='__main__':

    '''
    dataSet,labels=createDataSet()

    print dataSet
    print chooseBestFeatureToSplit(dataSet)

    list=['123','123','23','123','32','23']
    print majorityCnt(list)

    a=[[1,2,5],[3,4,7]]
    print len(a),len(a[0])


    myTree = testPlot.retrieveTree(0)
    print classify(myTree,labels,[1,0])
    print 'myTree:'
    testPlot.createPlot(myTree)

    print myTree
    '''
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    lensesTree = createTree(lenses,lensesLabels)

    print lensesTree

    testPlot.createPlot(lensesTree)