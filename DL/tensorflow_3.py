# 2020-2-15 tensorflow深度学习第四章
import tensorflow as tf
import numpy as np

# tensorflow基础
# 数值类型：标量Scalar:shape 为[] ， 向量Vector:shape 为[n]
# 矩阵Matrix:shape 为[𝑛, 𝑚]， 张量Tensor
scalar = tf.constant(1.2)
print(type(scalar))
print(scalar.numpy())  # Tensor转变为numpy格式

# 与标量不同，向量的定义必须通过List容器传给constant
vector = tf.constant([1,2,3.])  # 创建三个元素的向量
print(vector.shape)

matrix = tf.constant([[1,2],[3,4]])  # 创建二行二列的矩阵
tensor = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])  # 创建 3 维张量

# 字符串类型
# tf.strings模块中，提供了常见函数，小写化lower()，拼接join，长度length，切分split等
string = tf.constant('Hello')  # 创建字符串
print(tf.strings.upper(string))

# 布尔类型
boolen = tf.constant(True)  # 创建布尔类型标量
vectorBoolen = tf.constant([True, False])  # 创建布尔类型向量
print(boolen.numpy(),'\n',vectorBoolen)
# 需要注意的是，TensorFlow 的布尔类型和 Python 语言的布尔类型并不等价，不能通用
a = tf.constant(True)  # 创建 TF 布尔张量
print(a is True)  # TF 布尔类型张量与 python 布尔类型比较，False对象不等价
print(a == True)  # 仅数值比较

# 数值精度
# 常用的精度类型有 tf.int16,tf.int32,tf.int64,tf.float16,tf.float32,tf.float64等
# 其中 tf.float64 即为 tf.double.指定dtype即可:dtype=tf.int16
print(np.pi)  # pi tf.double
print(tf.constant(np.pi, dtype=tf.float32))  # 转化为float32

# 读取精度  tensor.dtype  与类型转换 tf.cast
# pi = np.pi 这种情况下会变成原python数据类型，所以不能之间运算
# 对于某些只能处理指定精度类型的运算操作，需要提前检验输入张量的精度类型，
# 并将不符合要求的张量进行类型转换
pi = tf.constant(np.pi)
if pi.dtype != tf.float32:
    pi = tf.cast(a, tf.float32)  # cast进行精度转换
print("pi_float32: ",pi)
int2bool = tf.constant([-1, 0, 1, 2])
tf.cast(int2bool, tf.bool)  # 整型转布尔类型

# 待优化张量
print("待优化张量")
"""
为了区分需要计算梯度信息的张量与不需要计算梯度信息的张量，TensorFlow 增加了
一种专门的数据类型来支持梯度信息的记录：tf.Variable。tf.Variable 类型在普通的张量类
型基础上添加了 name，trainable 等属性来支持计算图的构建。由于梯度运算会消耗大量的
计算资源，而且会自动更新相关参数，对于不需要的优化的张量，如神经网络的输入𝑿，
不需要通过 tf.Variable 封装；相反，对于需要计算梯度并优化的张量，如神经网络层的𝑾 
和𝒃，需要通过 tf.Variable 包裹以便 TensorFlow 跟踪相关梯度信息。
通过 tf.Variable()函数可以将普通张量转换为待优化张量
其中张量的 name 和 trainable 属性是 Variable 特有的属性，name 属性用于命名计算图中的
变量，这套命名体系是 TensorFlow 内部维护的，一般不需要用户关注 name 属性；trainable
属性表征当前张量是否需要被优化，创建 Variable 对象时是默认启用优化标志，可以设置
trainable=False 来设置张量不需要优化。
"""
a = tf.constant([-1, 0, 1, 2])  # 创建 TF 张量
aa = tf.Variable(a)  # 转换为 Variable 类型
print(aa.name, aa.trainable)  # Variable 类型张量的属性
a = tf.Variable([[1,2],[3,4]])  # 直接创建 Variable 张量

# 待优化张量可视为普通张量的特殊类型，普通张量其实也可以通过
# GradientTape.watch() 方法临时加入跟踪梯度信息的列表，从而支持自动求导功能。

# 创建张量-从数组、列表对象创建
print("创建张量")
"""
Numpy Array 数组和 Python List 列表是 Python 程序中间非常重要的数据载体容器，很
多数据都是通过 Python 语言将数据加载至 Array 或者 List 容器，再转换到 Tensor 类型
通过 TensorFlow 运算处理后导出到 Array 或者 List 容器，方便其他模块调用。
通过 tf.convert_to_tensor 函数可以创建新 Tensor，并将保存在 Python List 对象或者
Numpy Array 对象中的数据导入到新 Tensor 中
"""
list2tensor = tf.convert_to_tensor([1,2.])  # 从列表创建张量
array2tensor = tf.convert_to_tensor(np.array([[1,2.],[3,4]]))  # 从数组中创建张量
# 实际上，tf.constant()和 tf.convert_to_tensor()都能够自动的把 Numpy
# 数组或者 Python  列表数据类型转化为 Tensor 类型

# 创建全 0 或全 1 张量
"""
将张量创建为全 0 或者全 1 数据是非常常见的张量初始化手段。考虑线性变换
𝒚 = 𝑾𝒙 + 𝒃，将权值矩阵𝑾初始化为全 1 矩阵，偏置 b 初始化为全 0 向量，此时线性变化
层输出𝒚 = 𝒙，因此是一种比较好的层初始化状态。通过 tf.zeros()和 tf.ones()即可创建
任意形状，且内容全 0 或全 1 的张量。例如，创建为 0 和为 1 的标量
"""
print(tf.zeros([]),'\n',tf.ones([]))  # 创建全 0，全 1 的标量
print(tf.zeros([1]),tf.ones([1])) # 创建全 0，全 1 的向量
print(tf.zeros([2,2]))  # 创建全 0 矩阵，指定 shape 为 2 行 2 列
print(tf.ones([3,2]))  # 创建全 1 矩阵，指定 shape 为 3行2列

# 通过 tf.zeros_like, tf.ones_like 可以方便地新建与某个张量 shape 一致，
# 且内容为全 0 或 全 1 的张量。例如，创建与张量𝑨形状一样的全 0 张量：
a = tf.ones([2,3])  # 创建一个矩阵
print(tf.zeros_like(a))  # 创建一个与 a 形状相同，但是全 0 的新矩阵
# tf.*_like 是一系列的便捷函数，可以通过 tf.zeros(a.shape)等方式实现

# 创建自定义数值张量
"""
除了初始化为全 0，或全 1 的张量之外，有时也需要全部初始化为某个自定义数值的
张量，比如将张量的数值全部初始化为−1等。
通过 tf.fill(shape, value)可以创建全为自定义数值 value 的张量，形状由 shape 参数指
定。例如，创建元素为−1的标量
"""

fillOne =tf.fill([], -1) # 创建-1 的标量
vectorFillOne =tf.fill([1], -1) # 创建-1 的向量
matrixFillOne =tf.fill([2,2], 99) # 创建 2 行 2 列，元素全为 99 的矩阵

# 创建已知分布的张量
"""
正态分布(Normal Distribution，或 Gaussian Distribution)和均匀分布(Uniform 
Distribution)是最常见的分布之一，创建采样自这 2 种分布的张量非常有用，比如在卷积神
经网络中，卷积核张量𝑾初始化为正态分布有利于网络的训练；在对抗生成网络中，隐藏
变量𝒛一般采样自均匀分布。
通过 tf.random.normal(shape, mean=0.0, stddev=1.0)可以创建形状为 shape，均值为
mean，标准差为 stddev 的正态分布𝒩(mean, stddev2)。例如，创建均值为 0，标准差为 1
的正态分布

通过 tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.float32)可以创建
采样自[minval, maxval)区间的均匀分布的张量。例如创建采样自区间[0,1)，shape 为[2,2]
的矩阵
如果需要均匀采样整形类型的数据，必须指定采样区间的最大值 maxval 参数，
同时指 定数据类型为 tf.int*型
"""
normalTensor = tf.random.normal([2,2]) # 创建标准正态分布的张量
normal2 = tf.random.normal([2,2], mean=1,stddev=2) # 创建正态分布的张量,均值1标准差2
uniformTensor = tf.random.uniform([2,2]) # 创建采样自[0,1)均匀分布的矩阵
uniform2 = tf.random.uniform([2,2],maxval=10) # 创建采样自[0,10)均匀分布的矩阵'

# 创建序列
"""
在循环计算或者对张量进行索引时，经常需要创建一段连续的整型序列，可以通过
tf.range()函数实现。tf.range(limit, delta=1)可以创建[0, limit)之间，步长为 delta 的整型序
列，不包含 limit 本身。例如，创建 0~10，步长为 1 的整型序列
"""
range1 =tf.range(10)  # 0~10，不包含 10
range2 = tf.range(10,delta=2)  # 创建 0~10，步长为 2 的整形序列
range3 = tf.range(1,10,delta=2)  # 1~10
