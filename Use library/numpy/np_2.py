import numpy as np

# 索引和切片
a=np.array([[1,2,3],[1,5,4],[1,6,3],[1,7,3],[1,2,3]])
"""
print(a[1])  # 取一行
print(a[:,2])  # 取一列
print(a[0:2])  # 取多行
print(a[:,0:2])  # 取多列
print(a[[0,2],:2])  # 取不连续的多行 a[[]]并切片
print(a[:,[0,2]])  # 取不连续的多列
print(a[[1,3]])  # 取不连续的多行 a[[1, 3]]
"""
print(a>2)  # bool 索引
# a[a>2]=0  # 利用bool索引批量改变值
b=np.where(a<6,0,10)  # where是numpy的三元运算符，满足条件的变0，不满足的变10
print(b)

# clip 裁剪
c=a.clip(3,5)  # >5的变为5，小于3的变为3，其余不变
print(c)


# 数组的拼接
# 竖直拼接 vertically
np.vstack((a,b))
# 水平拼接 horizontally
np.hstack((a,b))

# 行列交换
a[[1,2],:]=a[[2,1],:]  # 行交换

# 构造全为0或者1的数组
np.zeros((a.shape[0]),1)  # 行和列
np.ones((a.shape[0]),1)
#创建一个对角线为1的方阵
np.eye(3)

# 获得最大最小值的位置
np.argmax(a,axis=0)
np.argmin(a,axis=1)

# numpy生成随机数
"""
.rand(d0,d1,..dn)  # 创建d0到dn维度的均匀分布的随机数数组，浮点数范围0-1
.randn(d0,d1,..dn)  # 创建d0到dn维度的标准正态分布随机数，浮点数，平均数0，标准差1
.randint(low,high,(shape))  # 给定上下范围选取随机整数，形状shape
.uniform(low,high,(size))  # 产生具有均匀分布的数组，low起始值，high结束值，size形状
.normal(loc,scale,(size))  # 从指定正态分布中随机抽取样本，分布中心loc（概率分布的均值），标准差scale，形状size
.seed(s)  # 随机数种子
"""

a=b  # 完全不复制，a，b相互影响
a=b[:]  # 视图的操作，a的数据完全由b保管，数据变化一致
a=b.copy()

# nan(NAN,Nan)  表示不是一个数字，读取文件有缺失会出现nan，不合适的计算（比如无穷大减无穷小）也会出现nan
# inf(-inf,inf) infinity inf表示正无穷，-inf负无穷
# 一个数字除以0会报错，可以用除以 np.inf
# 两个nan是不相等的  np.nan!=np.nan
# 判断nan的个数  np.count_nonzero(a!=a)
# 把nan替换为0  a[np.isnan(a)]=0
# nan和任何值计算都为nan

"""
常用统计计算,默认全部维度的结果
求和  a.sum(axis=None)
均值  a.mean(axis=None) 受离群点影响大
中值  np.median(a, axis=None)
最大  a.max(axis=None)
最小  a.min(axis=None)
极值  np.ptp(a, axis=None)  最大最小之差
标准差  a.std(axis=None)  越大越不稳定
方差 a.var()
np.exp(x)  e的x幂次方
"""