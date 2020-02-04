"""
如果是一维数据，我们可以用二叉查找树来进行存储，
但是如果是多维的数据，用传统的二叉查找树就不能够满足我们的要求了，
因此后来才发展出了满足多维数据的Kd-Tree数据结构

如何决定每次根据哪个维度对子空间进行划分呢？
直观的来看，我们一般会选择轮流来。先根据第一维，然后是第二维，然后第三……，那么到底轮流来行不行呢，
这就要回到最开始我们为什么要研究选择哪一维进行划分的问题。
我们研究Kd-Tree是为了优化在一堆数据中高频查找的速度，用树的形式，也是为了尽快的缩小检索范围，
所以这个“比对维”就很关键，通常来说，更为分散的维度，我们就更容易的将其分开，
是以这里我们通过求方差，用方差最大的维度来进行划分——这也就是最大方差法（max invarince）

如何选定根节点的比对数值呢？
选择何值未比对值，目的也是为了要加快检索速度。一般来说我们在构造一个二叉树的时候，
当然是希望它是一棵尽量平衡的树，即左右子树中的结点个数相差不大。所以这里用当前维度的中值是比较合理的。

BST的每个节点存储的是值，而Kd-Tree的根节点和中间节点存储的是对某个维度的划分信息，只有叶节点里才是存储的值


kd tree 采用的是从m 个样本的n维特征中，分别计算n 个特征的取值的方差
用方差最大的第k维特征n_k 来作为根节点
选择特征k_n 的取值的中位数n_kv对应的样本作为划分点，
对于所有第k维的取值小于n_kv的样本划入左子树，
对于所有第k维的取值大于n_kv的样本划入右子树，
对于左子树与右子树，采用和刚才同样的办法来找方差最大的特征来做根节点
递归的生成kd tree
"""

import numpy as np

# x = [[3,1,4],[2,3,7],[2,1,3],[2,4,5],[0,5,7],[1,4,4],[4,3,4],[6,1,4],[5,2,5],[4,0,6],[7,1,6]]
# x = np.array(x)
# var1_1 = np.cov(x[:,0])  # max = 4.19, select the first dimension
# var1_2 = np.cov(x[:,1])
# var1_3 = np.cov(x[:,2])
# print(var1_1, var1_2, var1_3)
# x = [[2,3,7],[2,1,3],[2,4,5],[0,5,7],[1,4,4]]
# x = np.array(x)
# var1_1 = np.cov(x[:,0])
# var1_2 = np.cov(x[:,1])
# var1_3 = np.cov(x[:,2])  # max = 2.56, select the third dimension
# print(var1_1, var1_2, var1_3)
# x = [[4,3,4],[6,1,4],[5,2,5],[4,0,6],[7,1,6]]
# x = np.array(x)
# var1_1 = np.var(x[:,0])  # max = 1.359, select the first dimension
# var1_2 = np.var(x[:,1])
# var1_3 = np.var(x[:,2])
# print(var1_1, var1_2, var1_3)


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

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()