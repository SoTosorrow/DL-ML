import pandas as pd
import numpy as np
import string
# pandas 能够处理数值(基于numpy)，也能处理其他类型的数据
# Series 一维，带标签数组
# DataFrame 二维，Series容器

a= pd.Series(np.arange(10),index=list(string.ascii_uppercase[:10]))
# print(a)
b={string.ascii_lowercase[i]:i for i in range(10)}
b=pd.Series(b)  # 通过字典创建Series
print(b)
print(b.index)  # 索引
print(b.values)  # 值
# Series对象本质上是两个数组构成，一个数组构成对象的键，一个构成值

# 读取外部数据
# c=pd.read_csv(".csv")
# 对于数据库的数据 pd.read_sql(sql_sentence, connection)

# DataFrame对象既有行索引，又有列索引
# index 即axis=0
# columns 即axis=1
d=pd.DataFrame(np.arange(12).reshape((3,4)),index=list("abc"),columns=list("wxyz"))
print(d)

"""
d.shape  行数列数
d.dtype  列数据类型
d.ndim  数据维度
d.index  行索引
d.columns  列索引
d.value  对象值，二维ndarrat数组

d.head(3)  显示头部几行，默认5
d.tail(3)  显示末尾几行，默认5
d.info()  相关信息概览，行数列数列索引列非空个数列类型内存占用
d.describe()  快速统合统计结果，计数均值标准差最大值四分位数最小值

d.sort_values(by="",ascending=True)  true升序false降序

选择某一行  a["a"]
选择行和列  a[:100]["a"]
"""

# 生成一段时间范围
# pd.date_range(start=None,end=None,periods=None,freq='D')
# start-end范围，频率freq，数量periods
print("data_time")
t =pd.date_range(start="20200209",end="20200311",periods=None,freq='D')
print(t)
"""
D day
B BusinessDay 每工作日
H hour
T min Minute
S Second
L ms 每毫秒
U 每微秒
"""