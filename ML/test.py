import numpy as np
test = [[1,2,3],
        [5,6,7],
        [7,8,9],
        [21,3,4]]
# npTest = np.array(test)
# subTest =npTest[:,2]
#
# print(subTest)
# finalTestList = []
# finalTestList=list(subTest)
# finalTestList.append(3)
# finalTestList=list(set(finalTestList))
# print(finalTestList)
# featureSet = list(set(subTest))
# print(featureSet)
test = np.array(test)
# print(test[:,1])
# print(test.shape)
# print(test.size)
# print(len(test[1]))

def splitDataSet(dataset,featureIndex):  # feature is attribute,featureIndex is index of attribute
    # calculate entropy of every feature or label
    # 效率低，ML library有改进版
    # 划分后的子集
    # 属性包含的值，比如性别：子集就是男和女
    # 将性别为男的数据行收集，女同样操作，再将性别这个属性数据去除
    subdataset=[]
    featureValues = dataset[:,featureIndex]
    print(featureIndex,featureValues)
    # 属性包含的数据种类，set去除重复
    featureSet = list(set(featureValues))
    print(featureSet)
    for i in range(len(featureSet)):
        newset=[]
        # 把每一个子集的数据放进去
        for j in range(dataset.shape[0]):
            if featureSet[i]==featureValues[j]:
                # 例如，假如性别为男，则收集数据newset
                newset.append(dataset[j,:])
                print(dataset[j,:])
        print(newset)
        newset =np.delete(newset,featureIndex,axis=1)
        print(newset)
        subdataset.append(np.array(newset))
        print(subdataset)
    return subdataset

splitDataSet(test, 1)