#y = theta0x0+theta1x1------>x0=1
import numpy as np
X = np.arange(0., 10., 0.2)
m = len(X)
y = 2 * X + 5 + np.random.randn(m)
#初始化参数
theta0 = theta1 = 0
#学习率
alpha = 0.00001
#迭代次数
cnt = 0
#误差
error0=error1 = 0
#指定一个阈值 用于检查两次误差的差  以便停止迭代
threshold = 0.0000001

while True:
    #梯度  diff[0]是theta0的梯度  diff[1]是theta1的梯度
    diff=[0,0]
    m = len(X)
    x0=1
    for i in range(0,m,2):
        diff[0] +=(y[i]-(theta0+theta1*X[i]))*x0
        diff[1] +=(y[i]-(theta0+theta1*X[i]))*X[i]
    theta0 = theta0+alpha*diff[0]
    theta1 = theta1+alpha*diff[1]
    #计算误差
    for i in range(m):
        error1+=(y[i]-(theta0+theta1*X[i]))**2
    error1/=m
    if abs(error1-error0)<threshold:
        break
    else:
        error0=error1
    cnt+=1
print(theta0,theta1,cnt)

def predict(theta0,theta1,x_test):
    return theta0+theta1*x_test
print(predict(theta0,theta1,15))
#87.14285714285715   87.29345695295272