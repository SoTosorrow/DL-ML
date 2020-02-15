import tensorflow as tf
import numpy as np
import os

# 分类问题
# load_data()函数返回两个元组(tuple)对象，第一个是训练集，第二个是测试集
(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()  # 加载 MNIST 数据集
# 转换为浮点张量，并缩放到-1~1 :先归一化，再变为-1~1
x = 2* tf.convert_to_tensor(x, dtype=tf.float32)/255. -1
y = tf.convert_to_tensor(y, dtype=tf.int32)  # 转换为整形张量
# y = tf.constant([0,1,2,3])   数字编码的 4 个样本标签
y = tf.one_hot(y, depth=10)  # one-hot 编码
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))  # 构建数据集对象
train_dataset = train_dataset.batch(512) # 批量训练

"""
图片的真实标签𝑦经过 One-hot 编码后变成长度为 10 的 非 0 即 1 的稀疏向量𝒚 ∈ {0,1}
10。预测模型采用多输入、多输出的线性模型 = 𝑾𝐓𝒙 + 𝒃，
其中模型的输出记为输入的预测值 ，我们希望 越接近真实标签𝒚越好。一般把输入经过
一次(线性)变换叫作一层网络。
"""

# 创建一层网络，设置输出节点数为 256，激活函数类型为 ReLU
tf.keras.layers.Dense(256, activation='relu')
# 利用 Sequential 容器封装 3 个网络层，前网络层的输出默认作为下一层的输入
model = tf.keras.Sequential([     # 3 个非线性层的嵌套模型
                tf.keras.layers.Dense(256, activation='relu'),  # 隐藏层 1
                tf.keras.layers.Dense(128, activation='relu'),  # 隐藏层 2
                tf.keras.layers.Dense(10)])  # 输出层，输出节点数为 10
# 第 1 层的输出节点数设计为 256，第 2 层设计为 128，输出层节点数设计为 10。
# 直接调用这个模型对象 model(x)就可以返回模型最后一层的输出𝑜

with tf.GradientTape() as tape: # 构建梯度记录环境
    # 打平操作，[b, 28, 28] => [b, 784]
    x = tf.reshape(x, (-1, 28*28))
    # Step1. 得到模型输出 output [b, 784] => [b, 10]
    out = model(x)
    # [b] => [b, 10]
    y_onehot = tf.one_hot(y, depth=10)  # 计算差的平方和，[b, 10]
    loss = tf.square(out-y_onehot)
    # 计算每个样本的平均误差，[b]
    loss = tf.reduce_sum(loss) / x.shape[0]
    # 再利用tf提供的自动求导函数 tape.gradient(loss, model.trainable_variables)
    # 求出模型中所有参数的梯度信息
    # Step3. 计算参数的梯度 w1, w2, w3, b1, b2, b3
    grads = tape.gradient(loss, model.trainable_variables)

    # 计算获得的梯度结果使用 grads 列表变量保存。再使用 optimizers 对象
    # 自动按照梯度更新法则去更新模型的参数𝜃。
    # 自动计算梯度
    grads = tape.gradient(loss, model.trainable_variables)
    # w' = w - lr * grad，更新网络参数
    tf.keras.optimizers.apply_gradients(zip(grads, model.trainable_variables))


