import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("car.csv",encoding="gbk")
x_train,x_test,y_train,y_test = train_test_split(df.values[:,:-1],df.values[:,-1],test_size=0.3,random_state=10)

def get_n_estimators():
    estimators = range(10,100)
    score = []
    for n_estimators in estimators:
        model = AdaBoostClassifier(n_estimators=n_estimators)
        model.fit(x_train,y_train)
        score.append(model.score(x_test,y_test))
    print(np.max(score))
    plt.plot(estimators,score)
    plt.show()
def get_learing_rate():
    learning_rate =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    score = []
    for i in learning_rate :
        model = AdaBoostClassifier(learning_rate =i)
        model.fit(x_train, y_train)
        score.append(model.score(x_test, y_test))
    print(np.max(score))
    plt.plot(learning_rate , score)
    plt.show()
def best():
    model = AdaBoostClassifier(n_estimators=70,learning_rate=0.3)
    model.fit(x_train, y_train)
    print(model.score(x_test,y_test))
if __name__=="__main__":
    # get_n_estimators()
    get_learing_rate()
    # best()