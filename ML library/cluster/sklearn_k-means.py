import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import pdist
from sklearn import metrics, preprocessing

data_path = 'data.csv'

# 读取数据文件
data_frame = pd.read_csv(data_path, encoding='gbk')

# 获取字段名
cols = list(data_frame.columns)


# 数据本身的散点图
def draw_scatter(x_label, y_label):
    # 绘图参数的设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.scatter(data_frame[x_label], data_frame[y_label])

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title('{0}-{1}散点图'.format(x_label, y_label))
    plt.show()


# K-Means聚类
def k_means_cluster(x_label, y_label, k):
    # 调用sklearn库生成K-Means模型
    #n_clusters聚集几个簇 max_iter最大迭代次数
    clu = KMeans(n_clusters=k, max_iter=300)
    #获取数据集
    X_value = data_frame[[x_label, y_label]].values
    print(type(X_value))
    # 开始进行K-Means聚类
    clu.fit(X_value)

    # 输出样本所属的簇
    print('样本所属簇编号:', clu.labels_)
    print(clu.labels_[100:200])
    # 输出簇中心坐标
    print('簇中心坐标:', clu.cluster_centers_)

    # 计算V值 （V=簇内平均误差平方和/簇间平均距离）
    v_value = clu.inertia_ / (k * np.average(pdist(clu.cluster_centers_)))
    print('v值{0}'.format(v_value))

    # 可视化聚类属性(散点图)

    # 参数设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 以簇编号作为颜色区分依据
    plt.scatter(data_frame[x_label], data_frame[y_label], c=clu.labels_)

    plt.title('K={0}聚类结果'.format(k))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# 思想：手肘法
# 随着聚类数k的增大，样本划分会更加精细，每个簇的聚合程度会逐渐提高，那么误差平方和SSE自然会逐渐变小。
# 并且，当k小于真实聚类数时，由于k的增大会大幅增加每个簇的聚合程度，故SSE的下降幅度会很大，而当k到达真实聚类数时，
# 再增加k所得到的聚合程度回报会迅速变小，所以SSE的下降幅度会骤减，然后随着k值的继续增大而趋于平缓，
# 也就是说SSE和k的关系图是一个手肘的形状，而这个肘部对应的k值就是数据的真实聚类数。当然，这也是该方法被称为手肘法的原因。

# 对比不同的K对V值的影响
def compare_k(x_label, y_label):

    k_ls = np.arange(1, 15)
    v_ls = []

    for k in k_ls:

        # 生成K-Means模型
        clu = KMeans(n_clusters=k, max_iter=300)

        X_value = data_frame[[x_label, y_label]].values
        # 开始进行K-Means聚类
        clu.fit(X_value)

        # 计算V值 （V=簇内平均误差平方和/簇间平均距离）
        v_value = clu.inertia_ / (k * np.average(pdist(clu.cluster_centers_)))

        # 添加到v_ls中
        v_ls.append(v_value)

    # 参数设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    print(v_ls)
    # 画k-v折线图
    plt.plot(k_ls, v_ls)

    plt.title('聚类个数对V值的影响')
    plt.xlabel('聚类个数')
    plt.ylabel('V值')
    plt.show()


# 归一化
def normalize_data(x_label, y_label, k):

    # 生成K-Means模型
    clu = KMeans(n_clusters=k, max_iter=300)

    # 归一化
    # scale_X = data_frame[[x_label, y_label]].apply(lambda x: x/x.max()).values
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(data_frame[[x_label, y_label]].values)
    # X_value = data_frame[[x_label, y_label]].values
    print(X_train_minmax)
    # 开始进行K-Means聚类
    clu.fit(X_train_minmax)
    # 输出样本所属的簇
    print('样本所属簇编号:', clu.labels_)
    # 输出簇中心坐标
    print('簇中心坐标:', clu.cluster_centers_)

    # 计算V值 （V=簇内平均误差平方和/簇间平均距离）
    v_value = clu.inertia_ / (k * np.average(pdist(clu.cluster_centers_)))
    print('v值{0}'.format(v_value))

    # 可视化聚类属性(散点图)

    # 参数设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 以簇编号作为颜色区分依据
    plt.scatter(X_train_minmax[:, 0], X_train_minmax[:, 1], c=clu.labels_)

    plt.title('K={0}聚类结果'.format(k))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

if __name__ == '__main__':
    pass
    # draw_scatter('当月MOU', '当月DOU')
    # k_means_cluster('当月MOU', '当月DOU', 3)
    compare_k('当月MOU', '当月DOU')
    # normalize_data('当月MOU', '当月DOU', 3)