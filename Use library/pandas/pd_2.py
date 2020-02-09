import pandas as pd
import numpy as np
import string

a={string.ascii_lowercase[i]:i for i in range(10)}
a=pd.Series(a)
print(a)
# print(a[2])
# print(a[2:10:2])  # step=2,start 2 to end 10
# print(a[a>5])
# print(a[["a","b"]])  # by the key
# print(a[[3,4,5]])  # by the index
# a[["a"]]=3

"""
a.loc  # 通过标签获得行数据索引
a.iloc  #  通过位置获取行数据
"""
print(a.loc["a"])
print(a.iloc[0])

d=pd.DataFrame(np.arange(12).reshape((3,4)),index=list("abc"),columns=list("wxyz"))
print(d)
print("loc1:\n",d.loc["a"])
print("loc2:\n",d.loc["a","w"])
print("loc3:\n",d.loc["a",["w","x"]])
print("loc4:\n",d.loc["b":,["x","z"]])
print("iloc:\n",d.iloc[1:2,1:3],"\n")
# d[d["w"]>0]=10
print(d[(d["w"]>=4) & (d["x"]>5)])  # 不同条件要用括号括起来，&且|或
# d[(d["row"].str.len()>4)&(d["count"]>700)

"""
字符串方法
cat  实现元素级的字符串连接操作，可指定分隔符
contains  返回表示各字符串是否含有指定模式的布尔型数组
count  模式的出现次数
endswith  相当于对各个元素执行 x.endswith(pattern)
startswith  
findall  计算各字符串的模式列表
get  获取各元素的第i个字符
join  根据指定的分隔符将Series中各元素的字符串连接起来
len  计算各字符串的长度
lower,upper  转换大小写
match  根据指定的正则表达式对每个元素执行re.match
pad  在字符串的左边，右边或两边添加空白字符
center  相当于pad(side='both')
repeat  重复值，例如 s.str.repeat(3)相当于对各个字符串执行x*3
replace  用指定字符串替换找到的模式
slice  对Series中的各个字符串进行子串截取
split  根据分隔符或者正则表达式对字符串进行拆分
strip,rstrip,lstrip  去除空白符，包括换行符
"""

# 判断数据是否为nan
pd.isnull(d)
pd.notnull(d)
# 处理：删除nan所在行列
d.dropna(axis=0,how='any',inplace=False)
# 处理：填充数据
d.fillna(d.mean())
d.fillna(0)
# 处理为0的数据，并不是每次0的数据都要处理，计算平均值时nan不参与计算，0参与
d[d==0]=np.nan

# d.join(d2) 默认情况是把行索引相同的数据合并到一起
# merge 按照指定的列把数据安装一定的方式合并到一起
# d.merge(d2,left_on="0,right_on="X") inner 并集 默认
# d.merge(d2,left_on="0,right_on="X",how="outer") outer 交集，nan补全
# d.merge(d2,left_on="0,right_on="X",how="left") left 左边为准，nan补齐
# d.merge(d2,left_on="0,right_on="X",how="right") right 右边为准，nan补齐

# pandas 分组操作
print("****"*5)
print(d.groupby(by="x"))