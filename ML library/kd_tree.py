"""
kd tree 采用的是从m 个样本的n维特征中，分别计算n 个特征的取值的方差
用方差最大的第k维特征n_k 来作为根节点
选择特征k_n 的取值的中位数n_kv对应的样本作为划分点，
对于所有第k维的取值小于n_kv的样本划入左子树，
对于所有第k维的取值大于n_kv的样本划入右子树，
对于左子树与右子树，采用和刚才同样的办法来找方差最大的特征来做根节点
递归的生成kd tree
"""

import numpy as np

x = [[2,3],[5,4],[9,6],[8,1],[7,2]]
a = np.array(x)

# 方差var和协方差cov
print(np.var(a[:,0]))
print(np.var(a[:,1]))
print(np.cov(a[:,0]))
print(np.cov(a[:,1]))

"""
三维样本6个，构建kd树的具体步骤为
1，找到划分的特征，6个数据点在x，y维度上的数据方差为6.91，5.37
所以在x轴上方差更大，用第一维特征建树。
2，确定划分点（7，2）。根据x维上的值将数据排序
6个数据的中值为7，所以划分点是（7，2），
这样，该节点的分割超平面就是通过（7，2）并垂直于：
划分点维度的直线x=7
3，确定左子空间和右子空间。分割超平面x=7将整个空间分为两部分
x<=7的部分为左子空间，包含三个节点（2，3）（5，4）（4，7）
右子空间包含（9，6）（8，1）
4，用同样的方法划分左子树的节点和右子树最终得到kd树
"""