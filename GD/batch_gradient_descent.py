# -*- coding: utf-8 -*-
from basic_functions import *


def bgd(x, y, w, rank):
    """
    use batch gradient descent method to calculate the parameter omega
    :param x: the x coordinate of the training sample, which is a row vector [x1, x2, ..., xn]
    :param y: the y coordinate of the training sample, which is a row vector [y1, y2, ..., yn]
    :param w: the parameter omega, which is a row vector [w0, w1, ..., wn]
    :param rank: the rank of the polynomial function
    :return: omega
    """
    # set the learning rate
    alpha = 0.01

    # set the parameter in the regular term
    p_lambda = 0.0

    # set a parameter to judge if the error function(E(omega) in the pdf) has converged to minimum
    epsilon = 0.0000001

    w_new = w.copy()
    error_before = calc_error(x, y, w_new, rank, p_lambda)
    error_after = 0

    # the iteration number of batch gradient descent
    it_number = 0

    # generate a vandermonde matrix used in the loop
    van_matrix = np.tile(x, (rank + 1, 1))
    for r in range(rank + 1):
        van_matrix[r] = van_matrix[r] ** r
    van_matrix = van_matrix.transpose()

    while True:

        it_number += 1
        print(it_number, error_after)

        # the gradient below is actually the negative gradient at w_new
        gradient = (y - calc_fx(x, w_new, rank)).dot(van_matrix) - p_lambda * w_new
        w_new = w_new + alpha * gradient

        error_after = calc_error(x, y, w_new, rank, p_lambda)

        if error_before - error_after < epsilon:
            break
        else:
            error_before = error_after

    return w_new
