"""
--GBDT调参示例--
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.externals import joblib
import matplotlib.pylab as plt

train_path = 'data.csv'

# 读取数据文件
data_frame_train = pd.read_csv(train_path, encoding='gbk')

# 划分训练集和测试集的X，y
X_train, y_train = data_frame_train.values[:, :-1], data_frame_train.values[:, -1]


# 不调整参数的效果
def default_param():
    model_name="gbdt.model"
    import os
    gbclf=""
    if os.path.exists(model_name):
        gbclf = joblib.load(model_name)
    else:
        # 实例化模型
        gbclf = GradientBoostingClassifier(random_state=10)
        # 模型训练
        gbclf.fit(X_train, y_train)
        # 模型对测试集进行预测
        joblib.dump(gbclf,model_name)
    y_pre = gbclf.predict(X_train)   # 预测值
    y_prb_1 = gbclf.predict_proba(X_train)[:, 1]  # 预测为1的概率
    # 输出预测准确度
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pre))
    print("AUC Score: %f" % metrics.roc_auc_score(y_train, y_prb_1))


# 首先从步长(learning rate)和迭代次数(n_estimators)入手，将步长初始值设置为0.1，对迭代次数进行网格搜索。
def adjust_n_estimators():
    param_dic = {'n_estimators': range(10, 101, 10),"max_depth":range(3,10)}
    gscv = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                                 min_samples_leaf=20, max_depth=8, max_features='sqrt',
                                                                 subsample=0.8, random_state=10),
                            param_grid=param_dic, scoring='roc_auc', iid=False, cv=5)
    gscv.fit(X_train, y_train)

    # print('result:{0}'.format(gscv.cv_results_))
    # result_df = pd.DataFrame(gscv.cv_results_)
    # result_df.to_csv('C:/Users/Lenovo/Desktop/result.csv', index=False)
    print('best_params:{0}'.format(gscv.best_params_))
    print('best_score:{0}'.format(gscv.best_score_))
    # best_n_estimators = 10


# 迭代次数有了，接下来对决策树进行调参：
# 首先对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索
def adjust_depth_samples():
    param_dic = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 200)}
    # 迭代次数选10
    gscv = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, min_samples_leaf=20,
                                             max_features='sqrt', subsample=0.8, random_state=10),
                                             param_grid=param_dic, scoring='roc_auc', iid=False, cv=5)
    gscv.fit(X_train, y_train)
    print('best_params:{0}'.format(gscv.best_params_))
    print('best_score:{0}'.format(gscv.best_score_))
    # max_depth: 3, min_samples_split: 100


# 先定下深度为3，但min_samples_split和其它参数还有关联，接下来要和min_samples_leaf一起调参
def adjust_samples_leaf():
    param_dic = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(40,101,10)}
    gscv = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=10,max_depth=3,
                                         max_features='sqrt', subsample=0.8, random_state=10),
                                         param_grid = param_dic, scoring='roc_auc',iid=False, cv=5)
    gscv.fit(X_train,y_train)
    print('best_params:{0}'.format(gscv.best_params_))
    print('best_score:{0}'.format(gscv.best_score_))
    # min_samples_leaf: 90, min_samples_split: 800


# 用选出来的参数去训练数据
def best_param():
    # 实例化模型
    gbclf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, max_depth=8,min_samples_leaf=90,
                                       min_samples_split=800, max_features='sqrt', subsample=0.8, random_state=10)
    # 模型训练
    gbclf.fit(X_train, y_train)
    # 模型对测试集进行预测
    y_pre = gbclf.predict(X_train)   # 预测值
    y_prb_1 = gbclf.predict_proba(X_train)[:, 1]  # 预测为1的概率
    # 输出预测准确度
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pre))
    print("AUC Score: %f" % metrics.roc_auc_score(y_train, y_prb_1))

# todo:在现有参数基础上接着调整max_features, subsample, learning_rate
if __name__ == '__main__':
    pass
    default_param()     # 0.9285
    # adjust_n_estimators() # 10
    # adjust_depth_samples() # 3 100
    # adjust_samples_leaf()
    #best_param()