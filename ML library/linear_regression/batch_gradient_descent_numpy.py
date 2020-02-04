#!/usr/bin/python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
# 构造训练数据 这个数据不包括x0
x = np.arange(0., 10., 0.2)
#训练样本的个数
m = len(x)
#构造x0
x0 = np.full(m, 1.0)
#将x0和x拼接起来 组成完整的训练数据
input_data = np.vstack([x0, x]).T  # 将偏置b作为权向量的第一个分量
featuerNum = input_data.shape[1]
print(input_data)
#y_train 生成的y 其实就是input_data对应的标签
target_data = 2 * x + 5 + np.random.randn(m)

# 两种终止条件
loop_max = 1000000  # 最大迭代次数(防止死循环)
epsilon = 1e-5

# 初始化权值
theta = np.random.randn(featuerNum)

alpha = 0.00001  # 步长(注意取值过大会导致振荡即不收敛,过小收敛速度变慢)
#记录误差变量
error = np.zeros(featuerNum)
error0 = error1 = 0
count = 0  # 循环次数

while count < loop_max:
    count += 1
    diff = np.zeros(2)
    # for i in range(m):
    #     diff += (np.dot(theta, input_data[i]) - target_data[i]) * input_data[i]
    #theta = [0,0]  input_data[i] (1,1)
    diff = (np.dot(input_data,theta)-target_data).dot(input_data)
    theta = theta - alpha* diff # 注意步长alpha的取值,过大会导致振荡
    # theta = theta - 0.005 * sum_m      # alpha取0.005时产生振荡,需要将alpha调小

    # 判断是否已收敛
    # if np.linalg.norm(theta - error) < epsilon:
    #     break
    # else:
    #     error = theta
    for i in range(m):
        error1 += (np.dot(theta, input_data[i]) - target_data[i])**2
    error1/=m
    if abs(error1-error0)<epsilon:
        break
    else:
        error0 = error1
print('loop count = %d' % count, '\tw:', theta)

plt.plot(x, target_data, 'g*')
plt.plot(x, theta[1] * x + theta[0], 'r')
plt.show()
#2.16931695 2.39108681
#4.77097422 2.04359657
