# python3
# -*- coding:utf-8 -*-
'''
@project: 实验3
@author: Erbenner
@file: tree_show.py.py
@ide: PyCharm
@time: 2019-01-05 23:47:40
@e-mail: jblei@mail.ustc.edu.cn
'''

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
label_name = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']


def getNumLeafs(root, num):
    if root is None:
        return 0

    return 1 + getNumLeafs(root.left, num) + getNumLeafs(root.right, num)


def getTreeDepth(root, hi):
    if root is None:
        return 0

    return 1 + max(getTreeDepth(root.left, hi), getTreeDepth(root.right, hi))


# 设置画节点用的盒子的样式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
# 设置画箭头的样式    http://matplotlib.org/api/patches_api.html#matplotlib.patches.FancyArrowPatch
arrow_args = dict(arrowstyle="<-")


# 绘图相关参数的设置
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # annotate函数是为绘制图上指定的数据点xy添加一个nodeTxt注释
    # nodeTxt是给数据点xy添加一个注释，xy为数据点的开始绘制的坐标,位于节点的中间位置
    # xycoords设置指定点xy的坐标类型，xytext为注释的中间点坐标，textcoords设置注释点坐标样式
    # bbox设置装注释盒子的样式,arrowprops设置箭头的样式
    '''
    figure points:表示坐标原点在图的左下角的数据点
    figure pixels:表示坐标原点在图的左下角的像素点
    figure fraction：此时取值是小数，范围是([0,1],[0,1]),在图的左下角时xy是（0,0），最右上角是(1,1)
    其他位置是按相对图的宽高的比例取最小值
    axes points : 表示坐标原点在图中坐标的左下角的数据点
    axes pixels : 表示坐标原点在图中坐标的左下角的像素点
    axes fraction : 与figure fraction类似，只不过相对于图的位置改成是相对于坐标轴的位置
    '''
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, \
                            xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', \
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


# 绘制线中间的文字(0和1)的绘制
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算文字的x坐标
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]  # 计算文字的y坐标
    createPlot.ax1.text(xMid, yMid, txtString)


# 绘制树
def plotTree(myTree, parentPt, nodeTxt):
    # 获取树的叶子节点
    numLeafs = getNumLeafs(myTree, 0)
    # 获取树的深度
    depth = getTreeDepth(myTree, 0)
    # firstStr = myTree.keys()[0]
    # 获取第一个键名
    if myTree.feat is not None:
        firstStr = label_name[myTree.feat]
    else:
        firstStr = label_name[myTree.isleaf]
    # 计算子节点的坐标
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,plotTree.yoff)
    # 绘制线上的文字
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制节点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 获取第一个键值
    # 计算节点y方向上的偏移量，根据树的深度
    plotTree.yoff = plotTree.yoff - 1.0 / plotTree.totalD

    if myTree.left is None:
        # 更新x的偏移量,每个叶子结点x轴方向上的距离为 1/plotTree.totalW
        plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
        # 绘制非叶子节点
        plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(myTree.isleaf))
    else:
        plotTree(myTree.left, cntrPt,'小于' + str(myTree.featVal))
        plotTree(myTree.right, cntrPt, '大于' + str(myTree.featVal))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD


# 绘制决策树，inTree的格式为{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
def createPlot(inTree):
    # 新建一个figure设置背景颜色为白色
    fig = plt.figure(1, facecolor='white')
    # 清除figure
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 创建一个1行1列1个figure，并把网格里面的第一个figure的Axes实例返回给ax1作为函数createPlot()
    # 的属性，这个属性ax1相当于一个全局变量，可以给plotNode函数使用
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 获取树的叶子节点
    plotTree.totalW = float(getNumLeafs(inTree, 0))
    # 获取树的深度
    plotTree.totalD = float(getTreeDepth(inTree, 0))
    # 节点的x轴的偏移量为-1/plotTree.totlaW/2,1为x轴的长度，除以2保证每一个节点的x轴之间的距离为1/plotTree.totlaW*2
    plotTree.xoff = -0.5 / plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
