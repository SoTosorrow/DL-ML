#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'lrg'
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import tree
from sklearn.metrics import confusion_matrix,roc_curve,auc
import pydotplus
file = "data.csv"
dataset = pd.read_csv(file, encoding="gbk").values
x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:-1], dataset[:, -1], test_size=0.3)

clf1 = KNeighborsClassifier(n_neighbors=20, weights="distance")
# clf2 = LogisticRegression(penalty='l1',solver="liblinear")
clf2 = LogisticRegression(penalty='l2',solver="newton-cg")
clf3 = DecisionTreeClassifier(criterion="gini",splitter="best",max_depth=4,min_samples_split=500,min_samples_leaf=90)

clfs = [clf1,clf2,clf3]

L= ["KNN","logisticRegression","DecisionTree"]
i=0
# 输出混淆矩阵和ROC曲线
for clf in clfs:
    # 训练数据
    clf.fit(x_train, y_train)

    # 输出预测测试集的概率
    y_prb_i = clf.predict_proba(x_test)[:, 1]

    # 得到误判率、命中率、门限

    fpr, tpr, thresholds = roc_curve(y_test, y_prb_i)
    # print(thresholds)
    # 计算auc
    roc_auc = auc(fpr, tpr)


    # 对ROC曲线图正常显示做的参数设定
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 绘图
    plt.scatter(fpr, tpr,label=L[i])
    plt.plot(fpr, tpr, label='AUC = %0.2f' % (roc_auc))

    i+=1

# 设置x、y轴刻度范围
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.legend(loc='best')
# 绘制参考线
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('命中率')
plt.xlabel('误判率')
plt.show()
