import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def logistic_regression(train_x, train_y, w, alpha, p_lambda):
    max_num = 500

    for i in range(max_num):
        output = sigmoid(train_x * w)
        # calculate the gradient
        gradient = train_x.transpose() * (output - train_y) + p_lambda * w
        w = w - alpha * gradient
    return w
