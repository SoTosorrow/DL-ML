from sklearn.neighbors import KNeighborsClassifier
import numpy as np

if __name__ == "__main__":
    dataset = np.loadtxt("knn_data.txt",dtype=np.str, delimiter=",")
    x = dataset[:,:-1].astype(np.float)
    y = dataset[:,-1]
    # 建立模型
    model = KNeighborsClassifier(n_neighbors=5,  # the number of k, default is 5
                                 weights='uniform',  # default is uniform ,also be 'distance'(距离越远权重越小）
                                 algorithm='auto',  # auto, ball_tree, kd_tree, brute
                                 leaf_size=30,  # 叶子数，默认30
                                 p=2,
                                 metric='minkowski',  # 距离的度量，默认明可夫斯基，p=2时，明可夫斯基距离等于欧氏距离
                                 n_jobs=1  # 线程个数
                                 )
    # 训练模型
    model.fit(x,y)
    # 预测
    predict_data = [[2,2]]  # 输入数据是二维的，所以预测数据也要是二维的
    result = model.predict(predict_data)
    print(result)
    result_probability = model.predict_proba(predict_data)  # 结果分布的概率
    print(result_probability)
    print(model.score(x, y))

"""
　n_neighbors=5           

int 型参数    knn算法中指定以最近的几个最近邻样
本具有投票权，默认参数为5

　　weights='uniform'       

str参数        即每个拥有投票权的样本是按什么比重投票，
'uniform'表示等比重投票，'distance'表示按距离反比投票，
[callable]表示自己定义的一个函数，这个函数接收一个
距离数组，返回一个权值数组。默认参数为‘uniform’

　　algrithm='auto'           
str参数       即内部采用什么算法实现。有以下几种选择参数：
'ball_tree':球树、'kd_tree':kd树、
'brute':暴力搜索、'auto':自动根据数据的类型和结构选择合适的算法。
默认情况下是‘auto’。暴力搜索就不用说了大家都知道。具体前两种树型
数据结构哪种好视情况而定。KD树是对依次对K维坐标轴，以中值切分构造
的树,每一个节点是一个超矩形，在维数小于20时效率最高
ball tree 是为了克服KD树高维失效而发明的，其构造过程是以质心
C和半径r分割样本空间，每一个节点是一个超球体。一般低维数据用
kd_tree速度快，用ball_tree相对较慢。超过20维之后的高维数据
用kd_tree效果反而不佳，而ball_tree效果要好，具体构造过程及
优劣势的理论大家有兴趣可以去具体学习。

　　leaf_size=30               

int参数      基于以上介绍的算法，此参数给出了kd_tree或者
ball_tree叶节点规模，叶节点的不同规模会影响数的构造和搜索速度，
同样会影响储树的内存的大小。具体最优规模是多少视情况而定。

　　matric='minkowski'     

str或者距离度量对象   即怎样度量距离。默认是闵氏距离，
闵氏距离不是一种具体的距离度量方法，它可以说包括了其他距离度量方式，
是其他距离度量的推广，具体各种距离度量只是参数p的取值不同或者是否
去极限的不同情况，
具体大家可以参考这里https://wenku.baidu.com/view/d3a9acede009581b6bd9ebf3.html
　
　　p=2                         

int参数       就是以上闵氏距离各种不同的距离参数，默认为2，
即欧氏距离。p=1代表曼哈顿距离等等

　　metric_params=None                

距离度量函数的额外关键字参数，一般不用管，默认为None

　　n_jobs=1                

int参数       指并行计算的线程数量，默认为1表示一个线程，
为-1的话表示为CPU的内核数，也可以指定为其他数量的线程，
这里不是很追求速度的话不用管，需要用到的话去看看多线程。



fit()                     

训练函数，它是最主要的函数。接收参数只有1个，就是训练数据集，
每一行是一个样本，每一列是一个属性。它返回对象本身，即只是修
改对象内部属性，因此直接调用就可以了，后面用该对象的预测函数
取预测自然及用到了这个训练的结果。其实该函数并不是
KNeighborsClassifier这个类的方法，
而是它的父类SupervisedIntegerMixin继承下来的方法。



predict()               

预测函数   接收输入的数组类型测试样本，一般是二维数组，
每一行是一个样本，每一列是一个属性返回数组类型的预测结果，
如果每个样本只有一个输出，则输出为一个一维数组。如果每个样
本的输出是多维的，则输出二维数组，每一行是一个样本，每一列是一维输出。



predict_prob()      

基于概率的软判决，也是预测函数，只是并不是给出某一个样本
的输出是哪一个值，而是给出该输出是各种可能值的概率各是多少
接收参数和上面一样返回参数和上面类似，只是上面该是值的地方
全部替换成概率，比如说输出结果又两种选择0或者1，上面的预测
函数给出的是长为n的一维数组，代表各样本一次的输出是0还是1.
而如果用概率预测函数的话，返回的是n*2的二维数组，每一行代
表一个样本，每一行有两个数，分别是该样本输出为0的概率为多少，
输出1的概率为多少。而各种可能的顺序是按字典顺序排列，
比如先0后1，或者其他情况等等都是按字典顺序排列。


score()                

计算准确率的函数，接受参数有3个。 X:接收输入的数组类型测试样本，
一般是二维数组，每一行是一个样本，每一列是一个属性。y:X这些预测
样本的真实标签，一维数组或者二维数组。sample_weight=None,
是一个和X第一位一样长的各样本对准确率影响的权重，一般默认为None.
输出为一个float型数，表示准确率。内部计算是按照predict()函数计
算的结果记性计算的。其实该函数并不是KNeighborsClassifier这个类
的方法，而是它的父类KNeighborsMixin继承下来的方法。

kneighbors()          

计算某些测试样本的最近的几个近邻训练样本。接收3个参数。X=None：
需要寻找最近邻的目标样本。n_neighbors=None,表示需要寻找目标样
本最近的几个最近邻样本，默认为None,需要调用时给出。
return_distance=True:是否需要同时返回具体的距离值。
返回最近邻的样本在训练样本中的序号。其实该函数并不是
KNeighborsClassifier这个类的方法，而是它的父类KNeighborsMixin

"""