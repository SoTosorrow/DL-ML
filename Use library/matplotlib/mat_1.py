import matplotlib.pyplot as plt
import random

# 假设一天中每隔两个小时(range(2,26,2))的气温分别是
# [15,13,14.5,17,20,25,26,26,27,22,18,15]

# 数据在x轴的位置，是一个可迭代对象
x= range(2,26,2)  # 左闭右开 2-24h
# 数据在y轴的位置，是一个可迭代对象
y= [15,13,14.5,17,20,25,26,26,27,22,18,15]
# x轴和y轴一起组成所有坐标


# 设置图片大小
"""
figure图形图标的意思，这里指的是我们画的图
通过实例化一个figure并传递参数，能在后台自动使用该figure实例
在图像模糊的时候可以传入dpi参数，让图片更清晰 dpi: Dots Per Inch每英寸上点的个数
figsize为图像大小
"""
fig =plt.figure(figsize=(10,6), dpi=80)

# 保存到本地
# plt.savefig("./sig_size.png")  # 可以保存为svg矢量图，放大无锯齿

# 描述信息
plt.xlabel("time")
plt.ylabel("temperature",rotation=45)
plt.title("this is lazy")

# 调整x,y刻度间距
"""
plt.xticks(range(2,25))   如果是range(2,25)则更密，2-24每一点一个刻度
_xtick_labels =[i/2 for i in range(4,49)]   2-24每零点五一个刻度
_xtick_labels=_xtick_labels[::3]  列表取步长(间隔取值，每隔3取一个
"""
plt.xticks(x)  # 把x的每个值都作为x轴 刻度,是一个列表
plt.yticks(range(min(y),max(y)+1))


# 线条样式
# 标记出特殊点
# 添加水印

plt.plot(x,y)  # 传入x,y. 通过plot绘制出折线图
plt.show()  # 执行程序的时候展示图形

# 120分钟对应120个20-35的数
random.seed(10)  # 设置随机种子，让不同时候随机得到的结果一样
a =range(0,120)
b =[random.randint(20,35) for i in range(120)]
fig =plt.figure(figsize=(15,6), dpi=80)

# 调整x轴刻度成字符串
_x = list(a)
# _xtick_labels = ["time{}".format(i) for i in _x]
_xtick_labels =["10点{}".format(i) for i in range(60)]
_xtick_labels += ["11:{}".format(i) for i in range(60)]
# 取步长，数字和字符串一一对应，数据的长度一样
# rotation  刻度旋转度数
plt.xticks(_x[::10], _xtick_labels[::10],rotation=-45)

# 设置中文显示，mat默认不支持中文字符
"""
# fc-list 查看支持的字体 linux/mac
# fc-list :lang=zh 查看支持的中文（冒号前有括号
# matplotlib.rc 修改默认字体

import matplotlib
font={'family':'MicroSoft YaHei',
      'weight':'bold',
      'size':'larger'}
matplotlib.rc("font",**font)
matplotlib.rc("font",family='MicroSoft YaHei')  # windows不行

windows:
import matplotlib.font_manager as fm
my_font = fm.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")
plt.xticks(_x[::10], _xtick_labels[::10],fontproperties=my_font)
"""

plt.plot(a,b)
plt.show()


