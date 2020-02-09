# 2020-2-6

import matplotlib.pyplot as plt
import random

random.seed(10)
a =[]
for i in range(100):
    a.append(random.randint(10,200))

# 组数等于极差除以组距
bin_width =5  # 组距
num_bins =int((max(a)-min(a)) //bin_width)
print(num_bins)
# plt.hist(a, num_bins) 绘制直方图，传入数据与组别数
plt.figure(figsize=(20,8))
plt.hist(a, num_bins)
# plt.hist(a, [min(a)+i*bin_width for i in range(num_bins)])
# 可以传入一个列表，长度为组数，值为分组依据，当组距不均匀时候使用
# normed:bool 是否绘制频率分布直方图，默认绘制频率直方图
# plt.hist(a, num_bins, normal=1)


plt.xticks(list(range(min(a),max(a)))[::bin_width],rotation=-45)
plt.grid(True,linestyle="--",alpha=0.5)

plt.show()