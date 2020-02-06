"""
1.随机取k个中心点
2. 计算所有点到中心点的距离
    将所有点 分别放入 中心点所在的簇
        更新中心点
            如果中心点不变 结束迭代
    迭代
"""
import numpy as np
import matplotlib.pyplot as plt

#获取数据集
def loadDataSet(filename):
    return np.loadtxt(filename,delimiter=",",dtype=np.float)

#取出k个中心点
def initCenters(dataset,k):
    """
    返回的k个中心点
    :param dataset:数据集
    :param k:中心点的个数
    :return:
    """
    centersIndex =  np.random.choice(len(dataset),k,replace=False)
    return dataset[centersIndex]
#计算距离公式
def distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

#kmeans的核心算法
def kmeans(dataset,k):
    """
    返回k个簇
    :param dataset:
    :param k:
    :return:
    """
    #初始化中心点
    centers = initCenters(dataset,k)
    n,m = dataset.shape
    #用于存储每个样本属于哪个簇
    clusters = np.full(n,np.nan)
    #迭代 标志
    flag = True
    while flag:
        flag = False
        #计算所有点到簇中心的距离
        for i in range(n):
            minDist,clustersIndex = 99999999,0
            for j in range(len(centers)):
                dist = distance(dataset[i],centers[j])
                if dist<minDist:
                    #为样本分簇
                    minDist = dist
                    clustersIndex = j
            if clusters[i]!=clustersIndex:
                clusters[i]=clustersIndex
                flag = True
        #更新簇中心
        for i in range(k):
            subdataset = dataset[np.where(clusters==i)]
            centers[i] = np.mean(subdataset,axis=0)
    return clusters,centers

#显示
def show(dataset,k,clusters,centers):
    n,m = dataset.shape
    if m>2:
        print("维度大于2")
        return 1
    #根据簇不同 marker不同
    colors = ["r","g","b","y"]
    for i in range(n):
        clusterIndex = clusters[i].astype(np.int)
        plt.plot(dataset[i][0],dataset[i][1],color=colors[clusterIndex],marker="o")
    for i in range(k):
        plt.scatter(centers[i][0],centers[i][1],marker="s")
    plt.show()
if __name__=="__main__":
    dataset = loadDataSet("testSet.txt")
    clusters,centers = kmeans(dataset,4)
    show(dataset,4,clusters,centers)
