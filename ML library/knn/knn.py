"""
自定义实现knn
给定一组数据集，包括（x，y）
给定一个预测的样本，
通过knn来预测该样本属于那个分类
样本集：
    1，1，A
    1，2，A
    1.5，1.5，A
    3，4，B
    4，4，B
预测数据
    2，2
通过knn对该数据进行分类预测

算法实现思路
    1，计算预测样本和数据集中的距离
    2，将所有距离从小到大排序
    3，计算前k个最小距离的类别个数
    4，返回前k个最小距离中 个数最多的分类

"""

import numpy as np
import operator


def handle_data(dataset):
    """
    :param dataset:  sample
    :return:  return x and y
    """
    x = dataset[:, :-1].astype(np.float)
    y = dataset[:, -1]
    return x, y

def knn_classifier(k, dataset, input):
    """
    :param k:
    :param dataset:  all the data
    :param input:  data of predict
    :return:  class of predict data
    """
    x, y = handle_data(dataset)

    # 1，计算预测样本和数据集中的距离
    distance = np.sum((input - x)**2, axis=1)**0.5
    # 2，将所有距离从小到大排序
    sortDist = np.argsort(distance)  # 排序之后输出排序的下标
    print(sortDist)
    # 3，计算前k个最小距离的类别个数
    countLabel = {}
    for i in range(k):
        label = y[sortDist[i]]
        countLabel[label] = countLabel.get(label, 0)+1
    # 4，返回前k个最小距离中 个数最多的分类
    # maxCount = 0
    # label = None
    # for k,v in countLabel.items():
    #     if v > maxCount:
    #         maxCount = v
    #         label = k
    sortLabel = sorted(countLabel.items(), key=operator.itemgetter(1), reverse=True)
    return sortLabel[0][0]


if __name__ == "__main__":
    dataset = np.loadtxt("knn_data.txt", dtype=np.str, delimiter=",")  # txt load separate by ","
    print(dataset)

    # predict data
    predict = [2, 2]
    print(knn_classifier(3, dataset, predict))

