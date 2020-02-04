from sklearn.linear_model import LinearRegression
# X = [[4], [8], [5], [10], [12]]
# y = [20, 50, 30, 70, 60]
x_train = [[1, 0., 3], [1, 1., 3], [1, 2., 3], [1, 3., 2], [1, 4., 4]]
y_train = [95.364, 97.217205, 75.195834, 60.105519, 49.342380]
model = LinearRegression()
model.fit(x_train,y_train)
print(model.coef_)
print(model.intercept_)
print(model.score(x_train,y_train))