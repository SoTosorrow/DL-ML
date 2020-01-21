from sklearn.neighbors import KNeighborsClassifier
import numpy as np

if __name__ == "__main__":
    dataset = np.loadtxt("knn_data.txt",dtype=np.str, delimiter=",")
    x = dataset[:,:-1].astype(np.float)
    y = dataset[:,-1]
    # 建立模型
    model = KNeighborsClassifier(n_neighbors=5,  # the number of k, default is 5
                                 weights='uniform',  # default is uniform ,also be 'distance'(距离越远权重越小）
                                 algorithm='auto',  # auto, ball_tree, kd_tree, brute
                                 leaf_size=30,  # 叶子数，默认30
                                 p=2,
                                 metric='minkowski',  # 距离的度量，默认明可夫斯基，p=2时，明可夫斯基距离等于欧氏距离
                                 n_jobs=1  # 线程个数
                                 )
    # 训练模型
    model.fit(x,y)
    # 预测
    predict_data = [[2,2]]  # 输入数据是二维的，所以预测数据也要是二维的
    result = model.predict(predict_data)
    print(result)
    result_probability = model.predict_proba(predict_data)  # 结果分布的概率
    print(result_probability)