# tensorflow深度学习第四章 4.5.1开始
# 2020-2-16

import tensorflow as tf
import numpy as np

# 4.5.1 Scalar
"""
在 TensorFlow 中，标量最容易理解，它就是一个简单的数字，维度数为 0，shape 为 []。
标量的一些典型用途是误差值的表示、各种测量指标的表示，比如准确度(Accuracy，
简称 acc)，精度(Precision)和召回率(Recall)等

out = tf.random.uniform([4,10]) #随机模拟网络输出
y = tf.constant([2,3,2,0]) # 随机构造样本真实标签
y = tf.one_hot(y, depth=10) # one-hot 编码
loss = tf.keras.losses.mse(y, out) # 计算每个样本的 MSE
loss = tf.reduce_mean(loss) # 平均 MSE,loss 应是标量
print(loss)
"""
# 4.5.2 Vector
"""
向量是一种非常常见的数据载体，如在全连接层和卷积神经网络层中，偏置张量𝒃就
使用向量来表示。如图 4.2 所示，每个全连接层的输出节点都添加了一个偏置值，把所有
输出节点的偏置表示成向量形式：𝒃 = [𝑏1, 𝑏2]^T。
考虑 2 个输出节点的网络层，我们创建长度为 2 的偏置向量𝒃，并累加在每个输出节
点上：
注意到这里 shape 为[4,2]的𝒛和 shape 为[2]的𝒃张量可以直接相加，这是为什么呢？
我们将 在 Broadcasting 一节为大家揭秘。
# z=wx,模拟获得激活函数的输入 z z = tf.random.normal([4,2])
b = tf.zeros([2]) # 创建偏置向量
z = z + b # 累加上偏置向量

通过高层接口类 Dense()方式创建的网络层，张量𝑾和𝒃存储在类的内部，由类自动创
建并管理。可以通过全连接层的 bias 成员变量查看偏置变量𝒃，例如创建输入节点数为 4，
输出节点数为 3 的线性层网络，那么它的偏置向量 b 的长度应为 3，实现如下：
fc = layers.Dense(3) # 创建一层 Wx+b，输出节点为 3 # 通过 build 函数创建 W,b 张量，输入节点为 4
fc.build(input_shape=(2,4))
fc.bias # 查看偏置向量
可以看到，类的偏置成员 bias 为长度为 3 的向量，初始化为全 0，这也是偏置𝒃的默认初始
化方案。同时偏置向量𝒃的类型为 Variable，这是因为𝑾和𝒃都是待优化参数。
"""

# 4.5.3 Matrix
"""
矩阵也是非常常见的张量类型，比如全连接层的批量输入张量𝑿的形状为[𝑏, 𝑑in]，其
中𝑏表示输入样本的个数，即 Batch Size，𝑑in表示输入特征的长度。例如特征长度为 4，一
共包含 2 个样本的输入可以表示为矩阵：
x = tf.random.normal([2,4]) # 2 个样本，特征长度为 4 的张量
令全连接层的输出节点数为 3，则它的权值张量𝑾的 shape 为[4,3]，我们利用张量𝑿、𝑾和
向量𝒃可以直接实现一个网络层，代码如下:
w = tf.ones([4,3]) # 定义 W 张量
b = tf.zeros([3]) # 定义 b 张量
o = x@w+b # X@W+b 运算

其中𝑿和𝑾张量均是矩阵，上述代码实现了一个线性变换的网络层，激活函数为空。一般
地，𝜎(𝑿@𝑾 + 𝒃)网络层称为全连接层，在 TensorFlow 中可以通过 Dense 类直接实现，特
别地，当激活函数𝜎为空时，全连接层也称为线性层。我们通过 Dense 类创建输入 4 个节
点，输出 3 个节点的网络层，并通过全连接层的 kernel 成员名查看其权值矩阵𝑾：
fc = layers.Dense(3) # 定义全连接层的输出节点为 3
fc.build(input_shape=(2,4)) # 定义全连接层的输入节点为 4
fc.kernel # 查看权值矩阵 W

"""

# 4.5.4 3dTensor
"""
三维的张量一个典型应用是表示序列信号，它的格式是
𝑿 = [𝑏, sequence len, feature len]
其中𝑏表示序列信号的数量，sequence len 表示序列信号在时间维度上的采样点数或步数，
 feature len 表示每个点的特征长度。
考虑自然语言处理(Natural Language Processing，简称 NLP)中句子的表示，如评价句
子的是否为正面情绪的情感分类任务网络，如图 4.3 所示。为了能够方便字符串被神经网
络处理，一般将单词通过嵌入层(Embedding Layer)编码为固定长度的向量，比如“a”编码
为某个长度 3 的向量，那么 2 个等长(单词数量为 5)的句子序列可以表示为 shape 为[2,5,3] 
的 3 维张量，其中 2 表示句子个数，5 表示单词数量，3 表示单词向量的长度。我们通过
IMDB 数据集来演示如何表示句子，代码如下
# 自动加载 IMDB 电影评价数据集
(x_train,y_train),(x_test,y_test)=keras.datasets.imdb.load_data(num_words=10
000) # 将句子填充、截断为等长 80 个单词的句子
x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=80)
x_train.shape

可以看到 x_train 张量的 shape 为[25000,80]，其中 25000 表示句子个数，80 表示每个句子
共 80 个单词，每个单词使用数字编码方式表示。我们通过 layers.Embedding 层将数字编码
的单词转换为长度为 100 个词向量：
# 创建词向量 Embedding 层类
embedding=layers.Embedding(10000, 100) # 将数字编码的单词转换为词向量
out = embedding(x_train)
out.shape
可以看到，经过 Embedding 层编码后，句子张量的 shape 变为[25000,80,100]，其中 100 表
示每个单词编码为长度是 100 的向量。

"""

"""
这里只讨论三、四维张量，大于四维的张量一般应用的比较少，如在元学习(Meta 
Learning)中会采用五维的张量表示方法，理解方法与三、四维张量类似，不再赘述。 四维张量在卷积神经网络中应用非常广泛，它用于保存特征图(Feature maps)数据，格
式一般定义为
[𝑏, ℎ, , 𝑐]
其中𝑏表示输入样本的数量，ℎ/ 分别表示特征图的高/宽，𝑐表示特征图的通道数，部分深
度学习框架也会使用[𝑏, 𝑐, ℎ, ]格式的特征图张量，例如 PyTorch。图片数据是特征图的一
种，对于含有 RGB 3 个通道的彩色图片，每张图片包含了ℎ行 列像素点，每个点需要 3
个数值表示 RGB 通道的颜色强度，因此一张图片可以表示为[ℎ, , 3]。如图 4.4 所示，最
上层的图片表示原图，它包含了下面 3 个通道的强度信息。

神经网络中一般并行计算多个输入以提高计算效率，故𝑏张图片的张量可表示为
[𝑏, ℎ, , 3]，例如
# 创建 32x32 的彩色图片输入，个数为 4 x = tf.random.normal([4,32,32,3])
# 创建卷积神经网络
layer = layers.Conv2D(16,kernel_size=3)
out = layer(x) # 前向计算
out.shape # 输出大小

其中卷积核张量也是 4 维张量，可以通过 kernel 成员变量访问：
layer.kernel.shape # 访问卷积核张量
TensorShape([3, 3, 3, 16])
"""