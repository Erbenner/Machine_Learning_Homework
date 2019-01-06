# python3
# -*- coding:utf-8 -*-
'''
@project: 实验3
@author: Erbenner
@file: id3.py
@ide: PyCharm
@time: 2019-01-04 23:54:15
@e-mail: jblei@mail.ustc.edu.cn
'''

# 使用决策树的ID3算法解决iris数据集上样本分类问题

from math import log


class Node:

    def __init__(self, feat=None, featVal=None, isleaf=-1, left=None, right=None):
        self.left = left
        self.right = right
        self.isleaf = isleaf    # int, -1 表明这是非叶节点，0~2表明是叶节点，对应的为类编号
        self.feat = feat        # int, 按照特征几来划分
        self.featVal = featVal  # float, 对于特征feat来说，从featVal处进行二分


def load_dataset():
    from sklearn import datasets
    iris_data = datasets.load_iris()
    data = iris_data.data.tolist()
    label = iris_data.target.tolist()
    for i in range(len(data)):
        data[i].append(label[i])
    return data


# 计算当前子集内信息熵
def calcShannonEnt(dataSet):
    # 当前子集内样本个数
    numEntries = len(dataSet)

    # 字典，保存类别n所包含的样本个数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 计算熵
    shannonoEnt = 0.0
    for key in labelCounts:
        # 每一个类别占总子集数量的百分比, 如10个样本中有3个正例，则正例的prob为0.3
        prob = float(labelCounts[key]) / numEntries
        # Ent = sum( - p * log p ), 这是熵的定义
        shannonoEnt -= prob * log(prob, 2)
    return shannonoEnt


# 选出当前子集中类别数目最多的那个类别
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        else:
            classCount[vote] += 1
    '''
    对字典按照value值进行降序排序
        第一个参数是传入一个可迭代的对象，此处迭代对象的元素为: (key, value)，是一个元组
            key为int, 如 2 代表'花瓣长度', value为float, 如 2.120
        第二个参数是按照元组中第二个值的大小进行排序
        第三个参数是降序，默认false为升序
    '''
    # >>>  若classCount为空则如何？什么时候为空？  <<<    潜在BUG
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    # 返回排序结果中（排序结果为list，每一个元素为一个元组）第一个元组的第一个元素
    # 返回对象为int ,表明类别
    return sortedClassCount[0][0]


def splitDataSetForSeries(dataSet, axis, value):
    '''
    按照给定的数值，将数据集分为大于与小于两部分
    '''

    eltDataSet = []
    gtDataset = []
    for feat in dataSet:
        if feat[axis] <= value:
            eltDataSet.append(feat)
        else:
            gtDataset.append(feat)

    return eltDataSet, gtDataset


# 选出增益最大的特征作为划分中枢
def chooseBestFeatureToSplit(dataSet):
    # 得到特征的数量
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = -1
    bestFeature = -1
    bestMid = -1
    for i in range(numFeature):
        # 获取dataSet第i列的所有值，也即第i个特征的所有取值
        featList = [number[i] for number in dataSet]
        featList = set(sorted(featList))
        T_a = []
        sun = featList.pop()
        moon = featList.pop()
        while len(featList) > 0:
            T_a.append((sun + moon) / 2)
            sun = moon
            moon = featList.pop()

        for value in T_a:
            # 对于每一个二分点来说，先将其分为大于与小于value的两个子集D+与D-
            subDataSet = splitDataSetForSeries(dataSet, i, value)
            newEntropy = 0
            for subSet in subDataSet:
                # 如果划分点刚好处于最小值或最大值，则可能出现空集
                if len(subSet) == 0:
                    continue
                # 迭代D-与D+两个子集，计算信息增益
                prob = len(subSet) / float(len(subSet))
                newEntropy += prob * calcShannonEnt(subSet)
            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i  # 第i个特征
                bestMid = value  # 从取值为bestMid开始二分
    # 返回应当对第几个特征进行划分，以及从哪个点开始二分
    return bestFeature, bestMid, bestInfoGain


'''
C4.5算法中，使用[信息增益率]来作为选择依据

def chooseBestFeatureToSplit(dataSet):
    # 得到特征的数量
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = -1
    bestFeature = -1
    bestMid = -1
    for i in range(numFeature):
        # 获取dataSet第i列的所有值，也即第i个特征的所有取值
        featList = [number[i] for number in dataSet]
        featList = set(sorted(featList))
        T_a = None
        if len(featList) == 1:
            T_a = [featList.pop()]
        else:
            T_a = []
            sun = None
            moon = featList.pop()
            while len(featList) > 0:
                sun = moon
                moon = featList.pop()
                T_a.append((sun + moon) / 2)

        for value in T_a:
            # 对于每一个二分点来说，先将其分为大于与小于value的两个子集D+与D-
            subDataSet = splitDataSetForSeries(dataSet, i, value)
            newEntropy = 0
            splitInfo = 0.0
            for subSet in subDataSet:
                # 如果划分点刚好处于最小值或最大值，则可能出现空集
                if len(subSet) == 0:
                    continue
                # 迭代D-与D+两个子集，计算信息增益
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subSet)
                splitInfo -= prob * log(prob, 2)
            infoGain = (baseEntropy - newEntropy) / splitInfo
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i  # 第i个特征
                bestMid = value  # 从取值为bestMid开始二分
    # 返回应当对第几个特征进行划分，以及从哪个点开始二分
    return bestFeature, bestMid, bestInfoGain
'''


def createTree(dataSet, e):
    '''
    NTOE:
        >>  调用者需要保证dataSet不为空   <<

        e ：float, 阈值。信息增益小于该值，则不再进行划分。
            该参数影响模型的复杂度，进而影响准确率、树高以及是否过拟合。
            e越大，模型越复杂，容易过拟合。
    '''

    # 拿到所有数据集的分类标签
    classList = [example[-1] for example in dataSet]

    # 如果类别相同
    if classList.count(classList[0]) == len(classList):
        return Node(isleaf=classList[0])

    # 如果类别不统一

    # 先尝试划分
    bestFeat, midSeries, bestInfoGain = chooseBestFeatureToSplit(dataSet)

    # 如果信息增益小于某个阈值，则不划分
    # >>  如果需要采用预剪枝，则在这里判断。 <<
    if bestInfoGain < e:
        return Node(isleaf=majorityCnt(classList))

    # 确定划分，生成中间节点。
    # 意为，若特征feat的取值小于featVal，则下一步去root.left,反之去root.right
    root = Node(feat=bestFeat, featVal=midSeries)
    print("当前划分选择特征", bestFeat, "，划分点:", midSeries)
    # 分为左右两个子集，小于、大于。
    eltDataSet, gtDataSet = splitDataSetForSeries(dataSet, bestFeat, midSeries)

    # 递归生成左右子树
    # subTree 应为Node类型
    if len(eltDataSet) == 0:
        subTree = None
    else:
        subTree = createTree(eltDataSet, e)
    root.left = subTree

    if len(gtDataSet) == 0:
        subTree = None
    else:
        subTree = createTree(gtDataSet, e)
    root.right = subTree

    return root


def pre(root, data):
    acc = 0
    res = []
    for simple in data:
        curr = root
        while curr.isleaf == -1:
            feat = curr.feat
            if simple[feat] <= curr.featVal:
                curr = curr.left
            else:
                curr = curr.right
        res.append(curr.isleaf)
        if simple[-1] == curr.isleaf:
            acc += 1
    return acc / len(data), res


if __name__ == '__main__':
    data = load_dataset()
    myTree = createTree(data, 0.1)
    acc, predict = pre(myTree, data)
    print('准确率', acc)
    from tree_show import *
    createPlot(myTree)
