from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
# model = SVC(gamma=0.1)
# model.fit(X,y)
# print(model.predict([[1.5, 1.5]]))
dataset = np.loadtxt("testSet.txt")
x_train,x_test,y_train,y_test = train_test_split(dataset[:,:-1],dataset[:,-1],test_size=0.3,random_state=1)
model = SVC(C=6,gamma=100)
model.fit(x_train,y_train)
print(y_test==model.predict(x_test))
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))
