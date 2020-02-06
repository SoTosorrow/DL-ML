import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics, preprocessing
data_path = 'data.csv'

# 读取数据文件
data_frame = pd.read_csv(data_path, encoding='gbk')


# DBSCAN聚类
def dbscan_cluster(x_label, y_label):

    # 生成DBSCAN模型
    clu = DBSCAN(eps=1, min_samples=5)

    X_value = data_frame[[x_label, y_label]].values

    # 开始进行DBSCAN聚类
    clu.fit(X_value)
    # 输出样本所属的簇
    print('样本所属簇编号:', clu.labels_)

    # 可视化聚类属性(散点图)

    # 参数设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 以簇编号作为颜色区分依据
    plt.scatter(data_frame[x_label], data_frame[y_label], c=clu.labels_)

    plt.title('DBSCAN聚类结果')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


if __name__ == '__main__':
    dbscan_cluster('当月MOU', '当月DOU')