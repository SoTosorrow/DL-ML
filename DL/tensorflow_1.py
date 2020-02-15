import tensorflow as tf
import numpy as np

"""
# 先导测试
# 创建四个张量并赋值
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)

with tf.GradientTape() as tape:  #构建梯度环境
    tape.watch([w])  # 将w加入梯度跟踪列表
    # 构建计算过程，函数表达式
    y = a * w**2 + b * w + c

# 自动求导
[dy_dw] = tape.gradient(y, [w])
print(dy_dw)
"""

"""
循环进行100次采样，每次从均匀分布U(-10，10)中随机采样一个数据x
同时从均值为0，方差为0.1^2的高斯分布中随机采样噪声eps
根据真实模型生成y的数据并保存为numpy数组
通过循环计算函数梯度 d_f 优化参数的更新，公式： x' = x - n * d_f
    n用来缩放梯度向量，一般设为某较小的值，如0.01，0.001
"""
# 线性模型
# 采集数据

data = []  # 保存样本集的列表
for i in range(100):  # 循环采样100个点
    x = np.random.uniform(-10., 10.)  # 随机采样输入
    eps = np.random.normal(0., 0.01)  # 采样高斯噪声
    y = 1.477 * x + 0.089 + eps  # 得到模型输出
    data.append([x, y])  # 保存样本点
data = np.array(data)  # 转换为2d Numpy数组


# 计算误差
def mse(b, w, points):
    # 根据当前的w，b参数计算均方差损失
    totalError = 0
    for j in range(0, len(points)):  # 循环迭代所有点
        x = points[j, 0]  # 获得i号点的输入x
        y = points[j, 1]  # 获得i号点的输出y
        # 计算差的平方，并累加：输出 减去 预测值
        totalError += (y - (w * x +b)) **2
    # 累加误差求平均，得到均方差
    return totalError / float(len(points))

# 计算梯度
def step_gradient(b_current, w_current, points, lr):
     # 计算误差函数在所有点上的导数，并更新 w,b
     b_gradient = 0
     w_gradient = 0
     M = float(len(points))  # 总样本数
     for i in range(0, len(points)):
         x = points[i, 0]
         y = points[i, 1]
         # 误差函数对 b 的导数：grad_b = 2(wx+b-y)，参考公式(2.3)
         b_gradient += (2/M) * ((w_current * x + b_current) - y)
         # 误差函数对 w 的导数：grad_w = 2(wx+b-y)*x，参考公式(2.2)
         w_gradient += (2/M) * x * ((w_current * x + b_current) - y)
     # 根据梯度下降算法更新 w',b',其中 lr 为学习率
     new_b = b_current - (lr * b_gradient)
     new_w = w_current - (lr * w_gradient)
     return [new_b, new_w]

# 梯度更新
# 计算出误差函数在w和b处的梯度后，根据前提公式更新w和b的值
# 把对数据集的所有样本训练一次称为一个epoch，共循环迭代 num_iterations个epoch

def gradient_descent(points, start_b, start_w, lr, num_iterations):
    # 循环更新 w,b 多次
    b = start_b  # b 的初始值
    w = start_w  # w 的初始值
    # 根据梯度下降算法更新多次
    for step in range(num_iterations):
        # 计算梯度并进行一次更新
        b, w = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points)  # 计算当前的均方差，用于监控训练进度
        if step % 50 == 0:  # 打印误差和实时的 w,b 值
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    return [b, w]  # 返回最后一次的 w,b

def main():
    # 加载训练集数据，这些数据是通过真实模型添加观测误差采样得到的
    lr = 0.01  # 学习率
    initial_b = 0  # 初始化 b 为 0
    initial_w = 0  # 初始化 w 为 0
    num_iterations = 1000
    # 训练优化 1000 次，返回最优 w*,b*和训练 Loss 的下降过程
    [b, w] = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)  # 计算最优数值解 w,b 上的均方差
    print(f'Final loss:{loss}, w:{w}, b:{b}')


main()
