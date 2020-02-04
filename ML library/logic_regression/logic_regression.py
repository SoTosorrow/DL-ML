#y = theta0+theta1*x1+theta2x2
import numpy as np
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/(1+np.exp(-x))


def weights(x_train,y_train):
    m,n = x_train.shape
    #初始化参数
    theta = np.random.rand(n)
    # 学习率
    alpha = 0.001
    # 迭代次数
    cnt = 0
    max_iter=500000
    # 误差
    # error0=error1 = 0
    # 指定一个阈值 用于检查两次误差的差  以便停止迭代
    threshold = 0.01
    while cnt<max_iter:
        cnt += 1
        diff = np.full(n,0)
        for i in range(m):
            diff =(y_train[i]-sigmoid(theta.T@x_train[i]))*x_train[i]
            theta = theta + alpha * diff
        if (abs(diff) < threshold).all():
            break
    return theta

def predict(x_test,weights):
    if sigmoid(weights.T@x_test)>0.5:
        return 1
    else:
        return 0

if __name__ =="__main__":
    x_train = np.array([[1, 2.697, 6.254],
                        [1, 1.872, 2.014 ],
                        [1, 2.312, 0.812],
                        [1, 1.983, 4.990],
                        [1, 0.932, 3.920],
                        [1, 1.321, 5.583],
                        [1, 2.215,1.560],
                        [1,1.659,2.932],
                        [1,0.865,7.362],
                        [1,1.685,4.763],
                        [1,1.786,2.523]])
    # y[i] 样本点对应的输出
    y_train = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    dataset = np.loadtxt('data.txt')
    x_train = dataset[:,0:-1]
    y_train = dataset[:, -1]
    x0 = np.full(len(y_train),1)
    x_train = np.vstack([x0,x_train.T]).T

    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)
    weights = weights(x_train,y_train)
    y_pred=[]
    for i in range(len(y_test)):
        y_pred.append(predict(x_test[i],weights))
    print(y_test==y_pred)