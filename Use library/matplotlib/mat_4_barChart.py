# 2020-2-6

import matplotlib.pyplot as plt

a=['a','b','c','d','e','f','g']
b=[2,5,12,6,2,7,8]

plt.grid(alpha=0.5)
# bar绘制条形图
plt.bar(range(len(a)), b,
        width=0.5)
# 设置x轴刻度
plt.xticks(range(len(a)), a)  # 传数据和字符串
plt.show()

plt.grid(alpha=0.5)
# barh绘制横着的条形图
plt.barh(range(len(a)), b,
        height=0.45,
        color='orange')
plt.yticks(range(len(a)), a)  # 传数据和字符串
plt.show()


a=['first','second','third','fourth']
b_1=[15746,312,4497,319]
b_2=[12357,125,2312,432]
b_3=[1234,523,6234,174]
bar_width=0.2

x_1=list(range(len(a)))
x_2=[i+bar_width for i in x_1]
x_3=[i+bar_width*2 for i in x_1]

plt.figure(figsize=(20,8),dpi=80)
plt.bar(x_1, b_1, width=bar_width)
plt.bar(x_2, b_2, width=bar_width)
plt.bar(x_3, b_3, width=bar_width)
plt.xticks([i+ bar_width for i in x_1],a)
plt.show()
