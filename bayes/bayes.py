# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import os
import re
import feedparser

# 测试数据
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #1 is abusive, 0 not
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

# 创建字典列表 使用set去除重复单词
def createVocabList(dataSet):
    vocabSet = set([])
    for words in dataSet:
        #print words
        vocabSet = vocabSet | set(words)


    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
    return returnVec

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        returnVec[vocabList.index(word)] += 1
    return returnVec

# 计算P(B|Ai) 和 P(Ai) Ai:类别 tainMatrix每行的列数相同
def trainNBO(trainMatrix,trainCategory):
    numTrainDoc = len(trainCategory)

    numWords = len(trainMatrix[0])
    # 二分类 计算P(A=1)大小
    pAbusive = sum(trainCategory)/float(numTrainDoc)

    #将分子初始化为1 分母初始化为0 使得不出现0概率 方便P(B1|Ai)*P(B2|Ai)*...P(Bn|Ai)的运算
    p1Num = ones(numWords)
    p1Denom = 2.0
    p0Num = ones(numWords)
    p0Denom = 2.0
    # 计算 P(B|Ai)

    for i in range(numTrainDoc):
        if trainCategory[i]==1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])

        else :
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vec = p1Num/p1Denom
    p0Vec = p0Num/p0Denom

    return p0Vec,p1Vec,pAbusive

# p0Vec,p1Vec是log后结果 相加即是相乘
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p0= sum(vec2Classify*p0Vec)+log(1-pClass1)
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)

    if p0 > p1:
        return 0
    else :
        return 1




# 测试朴素贝叶斯分类器
def testingBayse():
    wordsList,listClasses = loadDataSet()
    myVocabList = createVocabList(wordsList)
    trainMat = []

    for words in wordsList:
        trainMat.append(setOfWords2Vec(myVocabList,words))
    p0Vec,p1Vec,pA = trainNBO(array(trainMat),array(listClasses))

    testEntry =['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0Vec,p1Vec,pA)

    testEntry =['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0Vec,p1Vec,pA)


def textParse(bigString):

    listTokens = re.split(r'\W*',bigString)
    return [token.lower() for token in listTokens if len(token)>0]

def spamTest():


    # 预处理
    classList = []
    docList = []
    fullText = []

    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)
        fullText.extend(wordList)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)
        fullText.extend(wordList)




    # 生成训练和测试集合 训练集和测试集分开
    trainingSet = range(50)

    testSet = []

    for i in range(10):
        randomIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randomIndex])
        del(trainingSet[randomIndex])

    trainMatrix = []
    trainClasses = []
    vocabList = createVocabList(docList)

    for index in trainingSet:
        trainMatrix.append(bagOfWords2VecMN(vocabList,docList[index]))
        trainClasses.append(classList[index])

    p0Vec,p1Vec,pA = trainNBO(array(trainMatrix),array(trainClasses))

    errorCount = 0
    for index in testSet:
        testVec = bagOfWords2VecMN(vocabList,docList[index])
        classLabel = classifyNB(testVec,p0Vec,p1Vec,pA)
        if classLabel != classList[index]:
            errorCount += 1
            print 'classification error',docList[index]

    print 'the error rate is :',float(errorCount)/len(testSet)


if __name__=='__main__':

    #spamTest()
    #testingBayse()


    '''
    test =[['love', 'my', 'dalmation','love'],['stupid', 'garbage']]
    print array(test)
    for word in test:
        print set(word)




    dataSet,classVec=loadDataSet()
    vocabList = createVocabList(dataSet)
    print [0]*5

    numsData = len(dataSet)
    trainMatrix = []

    for i in range(numsData):
        trainMatrix.append(setOfWords2Vec(vocabList,dataSet[i]))
    print trainMatrix
    p0Vec,p1Vec,pAbusive = trainNBO(trainMatrix,classVec)
    print p0Vec
    print p1Vec
    print pAbusive
    '''


