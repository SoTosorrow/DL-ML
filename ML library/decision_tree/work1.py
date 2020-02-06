#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Vision'
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import  confusion_matrix,roc_curve,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# 输出3个模型的ROC曲线
def plot_ROC(file_name):
    dataset = pd.read_csv(file_name, encoding='gbk')
    x_train, x_test, y_train, y_test = train_test_split(dataset.values[:, :-1], dataset.values[:, -1], test_size=0.3)
    model1 = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=300)
    model2 = LogisticRegression()
    model3 = KNeighborsClassifier()
    model_list = [model1, model2, model3]
    model_name=["decisionTree","logisticRegression","knn"]
    i=0
    # figure, ax, = plt.subplot()
    for model in model_list:
        model.fit(x_train, y_train)
        pre = model.predict(x_test)
        c_matrix = confusion_matrix(y_test, pre)    # 混淆矩阵
        tn, fp, fn, tp = c_matrix.ravel()
        proba = model.predict_proba(x_test)[:, 1]   # 预测值概率
        FPR, TPR, threshold = roc_curve(y_test, proba)
        # FPR_list.append(FPR)
        # TPR_list.append(TPR)
        # ROC画图
        plt.plot(FPR, TPR, marker='o',label=model_name[i])      # 注意 这里阈值是方法自动生成的
        i+=1
    plt.legend(loc="best")
    plt.show()

if __name__ == '__main__':
    file_name = 'data.csv'
    # out_image(file_name)
    # graph_ROC(file_name)
    plot_ROC(file_name)
