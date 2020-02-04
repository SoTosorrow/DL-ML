from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import numpy as np

dataset = np.loadtxt("data.txt")
x_train,x_test,y_train,y_test = train_test_split(dataset[:,0:-1],dataset[:,-1],test_size=0.3)

model = SGDClassifier(penalty="l2",alpha=0.001)
model.fit(x_train,y_train)
print(y_test==model.predict(x_test))