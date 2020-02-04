#y = theta0+theta1*x1+theta2x2
x_train = [[1, 0., 3], [1, 1., 3], [1, 2., 3], [1, 3., 2], [1, 4., 4]]
# y[i] 样本点对应的输出
y_train = [95.364, 97.217205, 75.195834, 60.105519, 49.342380]
#学习率
alpha = 0.0001
#参数
theta0=theta1=theta2=0
#误差
error0=error1=0
#阈值
threshold = 0.0000001
#样本个数
m = len(y_train)
#次数
cnt = 0
while True:
    diff=[0,0,0]
    for i in range(m):
        diff[0] += y_train[i] - (theta0 + theta1 * x_train[i][1]+theta2*x_train[i][2])* x_train[i][0]
        diff[1] += (y_train[i] - (theta0 + theta1 * x_train[i][1]+theta2*x_train[i][2])) * x_train[i][1]
        diff[2] += (y_train[i] - (theta0 + theta1 * x_train[i][1] + theta2 * x_train[i][2])) * x_train[i][2]
    theta0 = theta0 + alpha / m * diff[0]
    theta1 = theta1 + alpha / m * diff[1]
    theta2 = theta2 + alpha / m * diff[2]
    # 计算误差
    for i in range(m):
        error1 += (y_train[i] - (theta0 + theta1 * x_train[i][1] + theta2 * x_train[i][2])) ** 2
    error1 /= m
    if abs(error1 - error0) < threshold:
        break
    else:
        error0 = error1
    cnt += 1
    print("次数%s"%cnt,theta0, theta1, theta2)
print(theta0, theta1,theta2, cnt)


