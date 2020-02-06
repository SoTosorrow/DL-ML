import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("car.csv",encoding="gbk")
x_train,x_test,y_train,y_test = train_test_split(df.values[:,:-1],df.values[:,-1],test_size=0.3,random_state=10)

def default():
    dtrain = xgb.DMatrix(data=x_train,label=y_train)
    dtest = xgb.DMatrix(data=x_test,label=y_test)
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    model = xgb.train(param,dtrain)
    preds = model.predict(dtest)
    print(preds)
    l = [1 if i > 0.5 else 0 for i in preds]
    import numpy as np
    print(np.sum(l==y_test))
def example():
    dtrain = xgb.DMatrix('agaricus.txt.train')
    dtest = xgb.DMatrix('agaricus.txt.test')
    print(dtrain)
    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    num_round = 2
    bst = xgb.train(param, dtrain, num_round)
    # make prediction
    preds = bst.predict(dtest)
    print(preds)

if __name__=="__main__":
    default()