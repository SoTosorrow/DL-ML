from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
data_path = 'data.csv'

# 读取数据文件
data_frame = pd.read_csv(data_path, encoding='gbk')

# 获取字段名
cols = list(data_frame.columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_frame.values[:, :-1], data_frame.values[:, -1], test_size=0.3)


# 输出ROC曲线
def plot_roc():
    # 构建Bagging模型
    clf_bagging = BaggingClassifier()
    # 构建Adaboost模型
    clf_ada = AdaBoostClassifier()
    # 构建GBDT模型
    clf_gbdt = GradientBoostingClassifier()
    # 构建RandomForest模型
    clf_rf = RandomForestClassifier()
    clf_xgb = XGBClassifier()
    clf_svm = SVC()
    # 构建集成模型集合
    clfs = [clf_bagging, clf_ada, clf_gbdt, clf_rf,clf_xgb]
    # 模型名称列表
    names = ['bagging', 'Adaboost', 'GBDT', 'RandomForest','xgboost']

    # 各模型预测为1的概率
    prbs_1 = []

    for clf in clfs:

        # 训练数据
        clf.fit(X_train, y_train)

        # 输出混淆矩阵
        pre = clf.predict(X_test)

        # 输出预测测试集的概率
        y_prb_1 = clf.predict_proba(X_test)[:, 1]
        prbs_1.append(y_prb_1)

    for index, value in enumerate(prbs_1):
        # 得到误判率、命中率、门限
        fpr, tpr, thresholds = roc_curve(y_test, value)
        # 计算auc
        roc_auc = auc(fpr, tpr)

        # 对ROC曲线图正常显示做的参数设定
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        plt.plot(fpr, tpr, label='{0}_AUC = {1:.5f}'.format(names[index], roc_auc))

    plt.title('ROC曲线')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('命中率')
    plt.xlabel('误判率')
    plt.show()


if __name__ == '__main__':
    plot_roc()