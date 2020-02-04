from sklearn.linear_model.stochastic_gradient import SGDRegressor
x_train = [[1, 0., 3], [1, 1., 3], [1, 2., 3], [1, 3., 2], [1, 4., 4]]
y_train = [95.364, 97.217205, 75.195834, 60.105519, 49.342380]
model = SGDRegressor(max_iter=5000000,alpha=0.00001)#[ 45.71878249 -13.02758034   1.14608487]
model.fit(x_train,y_train)
print(model.coef_)
print(model.intercept_)