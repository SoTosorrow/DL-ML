import numpy as np
test = [[1,2,3],
        [5,6,7],
        [7,8,9]]
npTest = np.array(test)
subTest =npTest[:,2]

print(subTest)
finalTestList = []
finalTestList=list(subTest)
finalTestList.append(3)
finalTestList=list(set(finalTestList))
print(finalTestList)
featureSet = list(set(subTest))
print(featureSet)