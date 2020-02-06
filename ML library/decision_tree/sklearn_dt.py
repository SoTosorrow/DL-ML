from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import tree
from sklearn.metrics import confusion_matrix,roc_curve,auc
import pydotplus
file = "data.csv"
dataset = pd.read_csv(file,encoding="gbk").values
x_train,x_test,y_train,y_test = train_test_split(dataset[:,:-1],dataset[:,-1],test_size=0.3)

#参数的调节
def getCriterion():
    criterions = ["gini","entropy"]
    for criterion in criterions:
        model = DecisionTreeClassifier(criterion=criterion)
        model.fit(x_train,y_train)
        print(criterion," training score:",model.score(x_train,y_train))
        print(criterion," testing score:",model.score(x_test,y_test))
#树的深度调节
def getDepth():
    max_depths=range(1,30)
    train_score =[]
    test_score = []
    for max_depth in max_depths:
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(x_train, y_train)
        train_score.append(model.score(x_train,y_train))
        test_score.append(model.score(x_test,y_test))
    print(train_score)
    print(test_score)
    plt.plot(max_depths,train_score,label="train",marker="*")
    plt.plot(max_depths,test_score,label="test",marker="o")
    plt.xlabel("max_depth")
    plt.ylabel("score")
    plt.legend(loc="best")
    plt.show()

#最小分裂点
def getMinSampleSplit():
    train_score = []
    test_score = []
    min_samples_split=range(100,1000,50)
    for min_samples in min_samples_split:
        model = DecisionTreeClassifier(min_samples_split=min_samples)
        model.fit(x_train, y_train)
        train_score.append(model.score(x_train, y_train))
        test_score.append(model.score(x_test, y_test))
    print(train_score)
    print(test_score)
    plt.plot(min_samples_split, train_score, label="train", marker="*")
    plt.plot(min_samples_split, test_score, label="test", marker="o")
    plt.xlabel("max_depth")
    plt.ylabel("score")
    plt.legend(loc="best")
    plt.show()

#叶子节点最小数
def getMinLeaf():
    train_score = []
    test_score = []
    min_samples_leaf = range(50,300,10)
    for min_samples in min_samples_leaf:
        model = DecisionTreeClassifier(min_samples_leaf=min_samples)
        model.fit(x_train, y_train)
        train_score.append(model.score(x_train, y_train))
        test_score.append(model.score(x_test, y_test))
    print(train_score)
    print(test_score)
    plt.plot(min_samples_leaf, train_score, label="train", marker="*")
    plt.plot(min_samples_leaf, test_score, label="test", marker="o")
    plt.xlabel("max_depth")
    plt.ylabel("score")
    plt.legend(loc="best")
    plt.show()
# 输出树形图
def out_image():
    # 模型初始化
    clf = DecisionTreeClassifier(max_depth=3)
    # 训练模型
    clf.fit(x_train, y_train)
    # 输出.dot文件
    tree.export_graphviz(clf, out_file=file.replace('.csv', '.dot'), filled=True, rounded=True)
    # #输出pdf/png
    # dot_data = tree.export_graphviz(clf,out_file=None,filled=True,rounded=True)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # # graph.write_pdf(file.replace('.csv', '.pdf'))
    # graph.write_png(file.replace('.csv', '.png'))
#roc曲线

def graph_roc():
    model = DecisionTreeClassifier(criterion="entropy",max_depth=5,min_samples_split=300)
    model.fit(x_train,y_train)
    #输出结果为1的概率
    proba1=model.predict_proba(x_test)[:,1]
    df1 = pd.read_csv(file, encoding="gbk")
    cols = list(df1.columns)[0:-1]
    df_test = pd.DataFrame(x_test, columns=cols)
    # 生成预测值和预测概率
    df_test.loc[:, '营销是否成功'] = y_test
    df_test.loc[:, '预测为1的概率'] = proba1
    if not os.path.exists("test.csv"):
        df_test.to_csv("test.csv",encoding="utf-8",index=False)

# 输出混淆矩阵和ROC曲线
def plot_roc():
    # 构建模型
    clf = DecisionTreeClassifier(max_depth=3)
    # 训练数据
    clf.fit(x_train, y_train)

    # 输出混淆矩阵
    pre = clf.predict(x_test)

    tn, fp, fn, tp = confusion_matrix(y_test, pre).ravel()
    # 更好的输出(二分类)
    print('tn={0},fp={1},fn={2},tp={3}'.format(tn,fp,fn,tp))
    # 输出预测测试集的概率
    y_prb_1 = clf.predict_proba(x_test)[:, 1]
    # 得到误判率、命中率、门限
    fpr, tpr, thresholds = roc_curve(y_test, y_prb_1)
    print(thresholds)
    # 计算auc
    roc_auc = auc(fpr, tpr)

    # 对ROC曲线图正常显示做的参数设定
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 绘图
    plt.scatter(fpr, tpr)
    plt.plot(fpr, tpr, 'g', label='AUC = %0.2f' % (roc_auc))
    plt.title('ROC曲线')
    # 设置x、y轴刻度范围
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.legend(loc='lower right')
    # 绘制参考线
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('命中率')
    plt.xlabel('误判率')
    plt.show()


if __name__=="__main__":
    # getCriterion()
    getDepth()
    # getMinSampleSplit()
    #getMinLeaf()
    # out_image()
    #graph_roc()
    # plot_roc()