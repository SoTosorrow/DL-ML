"""
--RandomForest调参示例--
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV, train_test_split

import matplotlib.pylab as plt

train_path = 'car.csv'

# 读取数据文件
data_frame_train = pd.read_csv(train_path, encoding='gbk')

# 划分训练集和测试集的X，y
X, y = data_frame_train.values[:, :-1], data_frame_train.values[:, -1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

# 不调整参数的效果(oob_score=True:采用袋外样本来评估模型的好坏,反映了模型的泛化能力)
def default_param():
    # 实例化模型
    rfclf = RandomForestClassifier(oob_score=True, random_state=10)
    # 模型训练
    rfclf.fit(X_train, y_train)
    # 模型对测试集进行预测
    y_pre = rfclf.predict(X_test)   # 预测值
    y_prb_1 = rfclf.predict_proba(X_test)[:, 1]  # 预测为1的概率
    # 输出oob_score以及auc
    print(rfclf.oob_score_)  # 0.8407142857142857
    print(rfclf.score(X_test,y_test))
    print("AUC Score: %f" % metrics.roc_auc_score(y_test, y_prb_1))


# 调节RandomForest最大决策树个数
def adjust_n_estimators():
    param_test1 = {'n_estimators': range(10, 71, 10),'max_depth': range(3, 14, 2)}
    gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                             min_samples_leaf=20, max_depth=8, max_features='sqrt',
                                                             random_state=10),
                            param_grid=param_test1, scoring='roc_auc', cv=5)
    gsearch1.fit(X_train, y_train)
    print('best params:{0}'.format(gsearch1.best_params_))
    print('best score:{0}'.format(gsearch1.best_score_))
    # best params:{'n_estimators': 70}
    #best params:{'max_depth': 5, 'n_estimators': 50}


# 再调节决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split
def adjust_depth_samples():
    param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(50, 201, 20)}
    gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=70,
                                                             min_samples_leaf=20, max_features='sqrt', oob_score=True,
                                                             random_state=10),
                            param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    gsearch2.fit(X_train, y_train)
    print('best params:{0}'.format(gsearch2.best_params_))
    print('best score:{0}'.format(gsearch2.best_score_))
    # best params:{'max_depth': 5, 'min_samples_split': 70}
    # best params:{'max_depth': 7, 'min_samples_split': 50}


# 目前模型下的袋外分数(是否提高--泛化能力)
def current_oob_score():
    rf1 = RandomForestClassifier(n_estimators= 70, max_depth=5, min_samples_split=70,
                                      min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10)
    rf1.fit(X_train,y_train)
    print(rf1.oob_score_)   # 0.8635714285714285


# min_samples_split和决策树其他的参数存在关联,需要内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。
def adjust_samples_leaf():
    param_test3 = {'min_samples_split': range(50, 150, 20), 'min_samples_leaf': range(10, 60, 10)}
    gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=70, max_depth=5,
                                                             max_features='sqrt', oob_score=True, random_state=10),
                            param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
    gsearch3.fit(X_train, y_train)
    print('best params:{0}'.format(gsearch3.best_params_))
    print('best score:{0}'.format(gsearch3.best_score_))
    # best params:{'min_samples_leaf': 10, 'min_samples_split': 80}
    #best params:{'min_samples_leaf': 10, 'min_samples_split': 50} 0.9346826586706646


# 对最大特征数进行调参
def adjust_max_features():
    param_test4 = {'max_features': range(2, 11,2)}
    gsearch4 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=70, max_depth=5, min_samples_split=80,
                                                             min_samples_leaf=10, oob_score=True, random_state=10),
                            param_grid=param_test4, scoring='roc_auc', iid=False, cv=5)
    gsearch4.fit(X_train, y_train)
    print('best params:{0}'.format(gsearch4.best_params_))
    print('best score:{0}'.format(gsearch4.best_score_))
    # best params:{'max_features': 3} {'max_features': 10}


# 最终模型的效果(泛化能力依旧不足？-->找更多数据！)
def best_params():
    rf2 = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=50,
                                 min_samples_leaf=10,max_features=10,oob_score=True, random_state=10)
    rf2.fit(X_train, y_train)
    print(rf2.oob_score_)  # 0.8571428571428571
    print(rf2.score(X_test,y_test))

if __name__ == '__main__':
    pass
    # default_param()
    # adjust_n_estimators()
    # adjust_depth_samples()
    # current_oob_score()
    # adjust_samples_leaf()
    # adjust_max_features()
    best_params()
