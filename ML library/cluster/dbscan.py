import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn import datasets
#距离计算公式
def distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

def dbscan(dataset,minPts,eps):
    """
    :param dataset:数据集
    :param minPts:最少点树
    :param eps:半径
    :return: 返回的是样本的簇的集合
    """
    n,m = dataset.shape
    #定义一个容器 用于存储样本的分类
    clusters = np.full(n,-1)
    #簇的类别
    k=-1
    for i in range(n):
        if clusters[i]!=-1:
            continue
        #获取领域中的所有样本点
        subdataset = [j for j in range(n) if distance(dataset[j],dataset[i])<=eps]
        if len(subdataset) < minPts:
            continue
        #建立簇的标记
        k+=1
        clusters[i] = k
        for j in subdataset:
            print(subdataset)
            clusters[j] = k
            if j>i:
                sub = [item for item in range(n) if distance(dataset[j],dataset[item])<=eps]
                if len(sub)>=minPts:
                    for t in sub:
                        if t not in subdataset:
                            subdataset.append(t)
    print(clusters)
    return clusters

X1, Y1 = datasets.make_circles(n_samples=2000, factor=0.6, noise=0.05,
                               random_state=1)
X2, Y2 = datasets.make_blobs(n_samples=500, n_features=2, centers=[[1.5,1.5]],
                             cluster_std=[[0.1]], random_state=5)

X = np.concatenate((X1, X2))
# plt.figure(figsize=(12, 9), dpi=80)
# plt.scatter(X[:,0], X[:,1], marker='.')
# plt.show()

C1 = dbscan(X, 10, 0.1)
print(C1)
plt.scatter(X[:, 0], X[:, 1], c=C1, marker='.')
plt.show()
