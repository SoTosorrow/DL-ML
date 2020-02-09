import numpy as np

# 创建数组,以下三种相同
a = np.array([1,2,3,4,5])
b = np.array(range(1,6))
c = np.arange(1,6)
# np.arange([start,] stop[,step,], dtype=None)
# 数组类名:type(a)  数据类型:a.dtype   数组形状:a.shape

# 修改数组数据类型
a=np.array([1,0,1,0],dtype=np.bool)  # 或者使用dtype='?'
print(a)
# 修改数组值得数据类型
a= a.astype(np.int32)
print(a)
# 修改浮点型得小数位数
c =np.arange(0.12345,3,0.23714)
c= np.round(c,2)
print(c)
# 修改数组形状
a=np.array([[1,2,3,5,1],[2,4,6,7,7],[5,2,1,2,7]])  # shape=(3,5)
a=a.reshape(5,3)
a=np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
a=a.reshape(a.shape[0]*a.shape[1],)  # a.flatter() flatten展开也是同样结果，变成一维
print(a)


#轴axis 在np中可以理解为方向，使用0,1,1....表示，对于一维数组只有一个0轴，2维数据有0和1轴
# 计算一个二维数组得平均值必须指定计算哪个方向上得平均值
d=np.arange(0,10).reshape(2,5)  # reshape中的2表示0轴长度，5为一轴长度，共2*5=10个数据
print(d)

# csv: Comma-Separated Value，逗号分隔符文件，换行和逗号分割行列的格式化文本，一行表示一条记录
"""
frame 文件字符串或者产生器，可以是.gz或者.bz2压缩文件
dtype  数据类型，以什么类型的数据读入数组，默认np.float
delimiter  分割字符串，默认是任何空格，改为逗号以对csv处理
skiprows  跳过前x行，一般跳过第一行表头
usecols   读取指定的列，索引，元组类型
unpack   如果True，读入属性将分别写入不同数组变量，False读入数据只写入一个数组，默认Flase
"""
# frame=".txt"
# np.loadtxt(frame,dtype="float32", delimiter=",", skiprows=0, usecols=None, unpack=False)

# 转置,对于np中的数组来说就是在对角线方向交换数据,三种方式实现转置，转置效果与交换轴方向一样
a.transpose()
# a.swapaxes(1,0)
a.T