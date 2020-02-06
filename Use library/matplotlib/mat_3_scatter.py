# 2020-2-6

import matplotlib.pyplot as plt

x_1=range(1,32)
x_2=range(51,82)
y_1=[11,17,16,11,12,11,12,6,6,7,8,9,12,15,14,17,18,21,16,17,20,14,15,15,15,19,21,22,22,22,23]
y_2=[26,26,28,19,21,17,16,19,18,20,20,19,22,23,17,20,21,20,22,15,11,15,5,13,17,10,11,13,12,13,6]

plt.figure(figsize=(15,8),dpi=80)

# 绘制散点图
plt.scatter(x_1,y_1)
plt.scatter(x_2,y_2)

# 调整x刻度
_x =list(x_1)+list(x_2)
_xtick_labels=['num:{}'.format(i) for i in x_1]
_xtick_labels +=['num:{}'.format(i-50) for i in x_2]

plt.xticks(_x[::3],_xtick_labels[::3],rotation=-45)

# 描述信息
plt.xlabel('time')
plt.ylabel('temperature',rotation=30)
plt.show()