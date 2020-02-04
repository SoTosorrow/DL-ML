# 2020-2-2

"""
决策树是从训练数据钟学习得出一个类似于流程图的树形结构
每个儿子节点表示在一个属性上的测试
每个叶子节点表示一种决策结果

决策树的生成包括构建（选择属性与顺序），和剪枝（消去噪点与孤立点）

决策树算法：
hunt
信息增益 information gain： id3
增益比率 gain ration： c4.5
基尼指数 Gini index ： CART
卡方检验 CHAID决策树
"""

"""
hunt ：每次从属性集中取出一个属性进行划分（每个只取一次，所以可能出现属性集用完了没划分完，
        这时按样本最多的数量的类别划分
    算法问题：没说如何选择最优划分属性，所以不用
输入： 训练集D={(x1,y1),(x2,y2),...(xm,ym)}
       属性集A={a1,a2,....ad}
过程： 函数 TreeGenerate(D,A)
1,生成节点 node （根节点）
2,if D 中样本全属于同一类别 C   then
3,      将node 标记为 C类叶节点 return
4,end if
5,if A= 空 OR D 中样本在A 上取值相同  then
6,      将node 标记为叶节点,其类别标记为 D中样本数最多的类  return
7,end if
8,从A 中选择最优化分属性 a*(算法中最重要的)
9,for a* 的每一个值 a_v do
10,     为node 生成一个分支 令 D_v 表示D 中在 a*上取值为 a_v 的样本子集
11,     if D_v 为空 then
12,             将分支节点标记为叶子节点，其类别标记为D 中样本最多的类  return
13,     else
14,             以TreeGenerate(D_v, A/ {a*}) 为分支节点
15,     end if
16, end for
输出： 以node为根节点的一棵决策树
"""



"""
id3: 如何选择最优算法---使用信息增益度选择测试属性
有改进版id4
1,决定分类属性集合
2,对目前的数据表，建立一个节点N
3,如果数据表中的数据都属于同一个类，N就是树叶，在树叶上标出所属的类（纯的类别）
4,如果数据表中没用其他属性可以考虑，则N也是树叶，按照少数服从多数的原则在树叶上标出所属类别（不纯的类别
5,否则，根据 平均信息期望 E  或者 GAIN 值选出一个最佳属性作为节点N的测试属性   （重点
6,节点属性选定后，对于该属性中每一个值：
    从N生成一个分支，并将数据表中与该分支有关的数据收集形成分支节点的数据表，
    在表中删除节点属性那一栏
7,如果分支数据表属性非空，则转1，运用以上算法从该节点建立子树


信息熵(Entropy)
是信息的度量，量化出信息的作用。信息量用bit度量，bit数和所有可能情况的对数有关
一条信息的信息量与它的不确定性有着直接的关系，变量的不确定性越大，熵就越大，搞清楚需要的信息量就越大
比如，要搞清楚一件非常不确定的事，或者是一无所知的事情，就需要了解大量信息，相反
    如果对某件事已经有了较多了解，那么不需要太多信息就能把它搞清楚
从这个角度看，信息量等于不确定性的多少

对于任意一个随机变量X，它的熵定义为：
    H(X) = -∑（P(x)  * log(P(x)) )   log中2为底
计算为：（D 划分为 1~n 个类 C_1，C_2......C_n）
    info(D) = -∑ p_i * log_pi    从1加到n, p_i为任意样本属于C_i 的概率，为 |C_(i,d)| / | D |

用熵衡量数据纯度或者混乱度:
正负例类包含 0% - 100%，不确定度先增大（50%最大，熵为1），再减小
所以数据变得越来越“纯”时，熵越来越小

按条件划分的信息熵：条件熵
假设按属性A划分D中的样本，且属性A根据训练数据的观测具有 V 个不同取值 {a1,a2,....aj,..,av}
如果A是离散值，可依属性A将D划分为V个子集{D1,D2,....Dj,...Dv}
其中，Dj为D中的样本子集，在A上具有属性aj
这些划分将对应于从该节点A出来的分支，按A对D划分后，数据集的信息熵：
    Info_A(D) = ∑[ (|Dj|/|D|) * Info(Dj) ]    从j=1到v
    其中，|Dj|/|D| 充当第j个划分的权重
    Info_A(D) 越小，表示划分纯度越高
信息增益 GAIN
gain(A) = Info(D) - Info_A(D)  数据集的熵 - 某一属性的熵
    选择熵最小，也就是信息增益最大
"""


# id3
import numpy as np

def createDataSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    dataSet = np.array([[0, 0, 0, 0, 'N'],
               [0, 0, 0, 1, 'N'],
               [1, 0, 0, 0, 'Y'],
               [2, 1, 0, 0, 'Y'],
               [2, 2, 1, 0, 'Y'],
               [2, 2, 1, 1, 'N'],
               [1, 2, 1, 1, 'Y']])
    labels = np.array(['outlook', 'temperature', 'humidity', 'windy'])  # 标签,属性
    return dataSet, labels

def createTestSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    testSet = np.array([[0, 1, 0, 0],
               [0, 2, 1, 0],
               [2, 1, 1, 0],
               [0, 1, 1, 1],
               [1, 1, 0, 1],
               [1, 0, 1, 0],
               [2, 1, 0, 1]])
    return testSet

# 计算信息熵
def dataset_entropy(dataset):
    print(dataset[:,-1])
    classLabel = dataset[:,-1]
    labelCount = {}  # dict
    for i in range(classLabel.size):
        label = classLabel[i]
        labelCount[label]=labelCount.get(label,0)+1
    # 熵值 entropy
    ent=0
    for k,v in labelCount.items():
        ent+=-v/classLabel.size*np.log2(v/classLabel.size)
    return ent

def splitDataSet(dataset,featureIndex):  # feature is attribute,featureIndex is index of attribute
    # calculate entropy of every feature or label
    # 划分后的子集
    subdataset=[]
    featureValues = dataset[:,featureIndex]
    featureSet = list(set(featureValues))
    for i in range(len(featureSet)):
        newset=[]
        for j in range(dataset.shape[0]):
            if featureSet[i]==featureValues[j]:
                newset.append(dataset[j,:])
        newset =np.delete(newset,featureIndex,axis=1)
        subdataset.append(np.array(newset))
    return subdataset
if __name__=="__main__":
    dataset,labels = createDataSet()
    # print(dataset_entropy(dataset))
    s = splitDataSet(dataset,0)
    for item in s:
        print(item)