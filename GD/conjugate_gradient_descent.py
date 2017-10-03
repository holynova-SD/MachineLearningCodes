# -*- coding: utf-8 -*-
from basic_functions import *


def cgd(x, y, w, rank):
    """
    use conjugate gradient descent method to calculate the parameter omega
    :param x: the x coordinate of the training sample, which is a row vector [x1, x2, ..., xn]
    :param y: the y coordinate of the training sample, which is a row vector [y1, y2, ..., yn]
    :param w: the parameter omega, which is a row vector [w0, w1, ..., wn]
    :param rank: the rank of the polynomial function
    :return: omega
    """
    # set the parameter in the regular term
    p_lambda = 0.0

    w_new = w.copy()

    # generate a vandermonde matrix used in the loop
    van_matrix = np.tile(x, (rank + 1, 1))
    for r in range(rank + 1):
        van_matrix[r] = van_matrix[r] ** r
    van_matrix = van_matrix.transpose()
    a_matrix = van_matrix.transpose().dot(van_matrix)

    for it_number in range(rank + 1):

        # the gradient below is actually the negative gradient at w_new
        gradient = (y - calc_fx(x, w_new, rank)).dot(van_matrix) - p_lambda * w_new

        if it_number == 0:
            d = gradient.copy()
        else:
            beta = (d.dot(a_matrix).dot(gradient.transpose())) / (d.dot(a_matrix).dot(d.transpose()))
            d = gradient - beta * d
        alpha = (gradient.dot(d.transpose())) / (d.dot(a_matrix).dot(d.transpose()))

        w_new = w_new + alpha * d

        error = calc_error(x, y, w_new, rank, p_lambda)

        print(it_number, error)

    return w_new






