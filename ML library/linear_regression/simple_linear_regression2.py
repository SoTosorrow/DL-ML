import numpy as np
class LinearRegression(object):

    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self,X,y):
        size = len(X)
        avgx = np.mean(X)
        avgy = np.mean(y)
        numerator = denominator = 0
        for i in range(size):
            numerator += (X[i] - avgx) * (y[i] - avgy)
            denominator += (X[i] - avgx) ** 2
        self.w = numerator / denominator
        self.b = avgy - self.w * avgx

    def predict(self,x_test):
        return self.w*x_test+self.b

if __name__=="__main__":
    X = [4,8,5,10,12]
    y = [20,50,30,70,60]

    model = LinearRegression()
    model.fit(X,y)
    print(model.predict(4))
