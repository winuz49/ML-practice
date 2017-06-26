# -*- coding: utf-8 -*-
import matplotlib.pyplot as plot



decisionNode = dict(boxstyle="sawtooth",fc="0.8")
leafNode = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeText,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeText,xy=parentPt,xycoords='axes fraction',xytext=centerPt,
                            textcoords='axes fraction',va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)


def createPlot_test():
    fig = plot.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1=plot.subplot(111,frameon=False)
    plotNode(U'决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode(U'叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plot.show()


def createPlot(inTree):
    fig = plot.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plot.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plot.show()

# 得到叶子总数
def getNumLeafs(myTree):

    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1




    return numLeafs

# 计算树的深度
def getTreeDepth(myTree):

    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]

    for key in secondDict.keys():

        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1+getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth=thisDepth

    return maxDepth

# 自定义的树的测试数据
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)



def plotTree(myTree, parentPt, nodeTxt):
    #if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)
    #this determines the x width of this tree
    depth = getTreeDepth(myTree)

    #plotTree.totalW=numLeafs
    #plotTree.totalD=depth
    firstStr = myTree.keys()[0]
    #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            #test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key],cntrPt,str(key))
            #recursion
        else:
            #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD



if __name__=='__main__':
    #createPlot_test()
    myTree = retrieveTree(1)
    print myTree
    print "number of leafs : %d" % getNumLeafs(myTree)
    print "depth of tree : %d" % getTreeDepth(myTree)
    createPlot(myTree)