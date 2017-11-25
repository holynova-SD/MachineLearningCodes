import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def newton(train_x, train_y, w):
    max_num = 20

    for i in range(max_num):
        output = sigmoid(train_x * w)
        # calculate the first derivative, also the gradient
        gradient = train_x.transpose() * (output - train_y)
        # calculate the hessian matrix
        a = output.dot((1 - output).transpose()).dot(np.eye(100))
        h = train_x.transpose() * a * train_x
        w = w - np.linalg.inv(h) * gradient
    return w
