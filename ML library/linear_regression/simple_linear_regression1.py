import numpy as np
#y = wx+b

def linear_regression_model(X,y):
    size = len(X)
    avgx = np.mean(X)
    avgy = np.mean(y)
    numerator = denominator = 0
    for i in range(size):
        numerator+= (X[i]-avgx)*(y[i]-avgy)
        denominator+= (X[i]-avgx)**2
    w =numerator/denominator
    b = avgy - w*avgx
    return w,b
def predict(w,b,x_test):
    return w*x_test+b

if __name__=="__main__":
    X = [4,8,5,10,12]
    y = [20,50,30,70,60]
    w,b = linear_regression_model(X,y)
    print(w,b)
    print(predict(w,b,15))#[87.14285714]
