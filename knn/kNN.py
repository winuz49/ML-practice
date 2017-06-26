# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import os

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0.1,0],[0,0]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):

    #行数即为个数
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffmat=diffMat**2
    sqDistance=sqDiffmat.sum(axis=1)
    #distance=sqDistance**0.5

    # 计算距离
    distances = ((diffMat**2).sum(axis=1))**0.5
    #根据距离 获得从小到大的排序的序号
    sortedDisIndex=distances.argsort()
    classCount={}
    for i in range(k):
        labelName=labels[sortedDisIndex[i]]
        classCount[labelName]=classCount.get(labelName,0)+1
    sortedClassCount=sorted(classCount.items(),key=lambda d:d[1],reverse=True)
    #print sortedClassCount
    return sortedClassCount[0][0]

# 将文件中的数据转换矩阵
def file2matrix(filename):
    fr = open(filename)
    allLines=fr.readlines()
    numbersOfLines=len(allLines)
    returnMat=zeros((numbersOfLines,3))
    labels=[]
    index=0
    for line in allLines:
        line=line.strip()

        listFromLine=line.split('\t')
        returnMat[index , :]=listFromLine[0:3]
        labels.append(int(listFromLine[-1]))
        index=index+1


    return returnMat,labels
#归一化: newValue=(value-minVlaue)/(maxValue-minValue)标准差的做法 还有方差的做法
def autoNorm(dataSet):

    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    ranges = maxValue-minValue
    m = dataSet.shape[0]

    normDataSet=(dataSet-tile(minValue,(m,1))) / (tile(ranges,(m,1)))
    #print normDataSet
    return normDataSet,ranges,minValue

def datingClassTest():
    dataSet,labels=file2matrix("datingTestSet.txt")
    normMat,ranges,minValue=autoNorm(dataSet)
    len = normMat.shape[0]
    hoRatio =0.10
    numTestVec = int(hoRatio*len)
    errorCount=0.0
    for i in range(numTestVec):
        classLabel=classify0(normMat[i,:],normMat[numTestVec:len,: ],labels[numTestVec:len],3)
        print 'the classifier came back with %d ,the real answer is : %d' % (classLabel,labels[i])
        if (classLabel!=labels[i]):
            errorCount += 1.0
    print "the total error rate is : %f"  % (errorCount/float(numTestVec))

def classifyPerson():

    resultList=['not at all','in small doses','in large doses']
    percentTats=float(raw_input('percentage of time spent playing video games?'))
    ffMiles = float(raw_input('frequent flier miles earned per year?'))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))
    dataSet,labels = file2matrix("datingTestSet.txt")
    normMat,ranges,minValue=autoNorm(dataSet)
    inX = array([percentTats,ffMiles,iceCream])
    label =int (classify0((inX-minValue)/ranges,normMat,labels,3))
    print resultList[label-1]
    return

#TODO 了解plot功能
def createFilstPlot(filename):

    datingDataMat,datingLabels=file2matrix(filename)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
    ax.axis([-2,25,-0.2,2.0])
    plt.xlabel('Percentage of Time Spent Playing Video Games')
    plt.ylabel('Liters of Ice Cream Consumed Per Week')
    plt.show()

def img2vec(filename):
    returnVec = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0,i*32+j]=lineStr[j]

    return returnVec

def hardWritingClassTest():

    hwLabels=[]

    trainingFileList = os.listdir("trainingDigits")
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))

    for i in range(m):
        filenameStr=trainingFileList[i]
        fileStr = filenameStr.split('.')[0]
        classNumber=int(fileStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:]=img2vec("trainingDigits/%s" % filenameStr)

    testFilelist = os.listdir("testDigits")
    mTest = len(testFilelist)
    errorCount=0.0
    for i in range(mTest):
        filenameStr=testFilelist[i]
        fileStr = filenameStr.split('.')[0]
        classNumber=int(fileStr.split('_')[0])
        testVec=img2vec("trainingDigits/%s" % filenameStr)
        testLabel=classify0(testVec,trainingMat,hwLabels,3)
        #print 'the classifier came back with %d ,the real answer is : %d' % (testLabel,classNumber)

        if (testLabel!=classNumber):
            print 'the classifier came back with %d ,the real answer is : %d' % (testLabel,classNumber)
            errorCount +=1.0

    print '\nthe total error is %d' % errorCount
    print '\nthe total error rate is %f' % (errorCount/float(mTest))

    return

if __name__=='__main__':


    hardWritingClassTest()
    group,labels=createDataSet();
    print classify0([0,0],group,labels,3)

    #createFilstPlot("datingTestSet.txt")
    #datingDataMat,datingLabels=file2matrix("datingTestSet.txt")

    b=np.array([ [1,2,3,4,5],[6,7,8,9,0],[2,3,4,11,1],[2,3,4,11,1] ])
    print b.min(0)
    print b.max(0)
    #autoNorm(file2matrix("datingTestSet.txt")[0])
    #datingClassTest()

    #classifyPerson()
    #files=os.listdir("testDigits")
    print os.path.exists("trainingDigits")
    #print os.listdir("trainingDigits")



    '''
    print 'hhhh'
    b=np.array([ [1,2,3,4,5],[6,7,8,9,0],[2,3,4,11,1],[2,3,4,11,1] ])
    print b

    print b.sum(axis=0)

    print b.sum(axis=1)
    a=[0,1]
    print tile(a,(2,1))
    group=array([[1.0,1.1],[1.0,1.0],[0.1,0],[0,0]])
    print group.shape

    for i in range(3):
        print i

    test=np.array(random.rand(4,4))
    print test
    print test.argsort()

    x = np.array([[0, 3], [2, 2]])
    print np.argsort(x,axis=0)
    print np.argsort(x,axis=1)

    classCount={}
    print type(classCount)

    dict = {"a" : "apple", "b" : "grape", "c" : "orange", "d" : "banana"}
    print dict

    print sorted(dict.items(),key=lambda d:d[0])
    print sorted(dict.items(),key=lambda d:d[1])
    '''