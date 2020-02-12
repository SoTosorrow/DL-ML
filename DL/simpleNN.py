import numpy as np

"""
layer0:输入层，输入x
w0:l0与l1之间的权重
layer1:中间层，4个神经元
w1:l1与l2之间的权重
layer2:输出层，输出01分类

"""
# 激活函数
def sigmoid(x, deriv=False):
    # deriv =True:计算导数, False 不计算导数（控制是前向还是反向传播
    if deriv is True:  # 反向传播，求导
        return x* (1-x)
    return 1/(1+np.exp(-x))  # 前向传播的值  e的-x幂次方

# 构造数据(5,3)
x= np.array(
    [[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1]])
# label(5,1)
y= np.array([[0],[1],[1],[0],[0]])
np.random.seed(1)

# 构造一个三行四列的随机矩阵,3是前一层的3，4是后面神经元的数量
# random: 0~1,    *2-1: -1~1
w0 =2 *np.random.random((3,4)) -1
# w1 前连中间层，后连输出层，所以前面对应4个神经元，后面准备输出分类的0或1，所以是1
w1 =2* np.random.random((4,1)) -1

# 迭代60000次
for i in range(60000):
    l0 =x
    l1 = np.dot(l0, w0)
    # 通过l1后经过激活函数
    l1 = sigmoid(l1)
    l2 = np.dot(l1, w1)
    l2 = sigmoid(l2)
    l2_error = y -l2  # 预测值和真实值之间差异值
    if (i %10000)== 0:
        print("Error" +str(np.mean(np.abs(l2_error))))
    # 反向传播
    l2_delta = l2_error * sigmoid(l2,deriv=True)  # 通过差异更新权重，差异越大更新越大
    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * sigmoid(l1,deriv=True)

    # y-l2:+=  ; l2-y:-=
    # 先更新w1
    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)

