# 2020-2-6
# 官方demo：https://matplotlib.org/gallery/index.html

import matplotlib.pyplot as plt
import random

x=range(11,31)
y_1=[1,0,1,1,2,4,3,2,3,4,4,5,6,5,4,3,3,1,1,1]
y_2=[1,0,3,1,2,2,3,3,2,1,2,1,1,1,1,1,1,1,1,1]

plt.figure(figsize=(15,8),dpi=80)
# 多条折线一起画,标记折线信息: plt多次plot,+label
"""
plt.plot(x, # x
         y, # y
         color ='r',  # 线条颜色
         linestyle='--',   # 线条风格 ':','-.','--','-',' '
         linewidth=5,      # 线条粗细
         alpha=0.5     # 透明度
         label='something'  # label
         )
自定义绘制图形风格
"""
plt.plot(x,y_1, label="self",
         linestyle=':')
plt.plot(x,y_2, label="another")

# 添加图例(每条线表示什么),结合plot的label
"""
loc=location
best 0
upper right 1 default
upper left 2
lower left 3
lower right 4
right 5       etc
"""
plt.legend(loc=10)  # prop=font


# 绘制网格
plt.grid(alpha=0.8,linestyle='--')
plt.show()