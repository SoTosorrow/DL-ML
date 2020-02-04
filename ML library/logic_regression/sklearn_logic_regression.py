from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

dataset = np.loadtxt("data.txt")
x_train,x_test,y_train,y_test = train_test_split(dataset[:,0:-1],dataset[:,-1],test_size=0.3)

model = LogisticRegression(penalty="l2")
model.fit(x_train,y_train)
print(y_test==model.predict(x_test))