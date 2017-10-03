# -*- coding: utf-8 -*-
import numpy as np


def calc_fx(x, w, rank):
    """
    calculate the result of f(x) which is a polynomial function
    :param x: the x coordinate of the training sample, which is a row vector [x1, x2, ..., xn]
    :param w: the parameter omega, which is a row vector [w0, w1, ..., wn]
    :param rank: the rank of the polynomial function
    :return: the result of the polynomial function, which is a row vector [f(x1), f(x2), ..., f(xn)]
    """
    van_matrix = np.tile(x, (rank + 1, 1))
    for r in range(rank + 1):
        van_matrix[r] = van_matrix[r] ** r
    fx = w.dot(van_matrix)
    return fx


def calc_error(x, y, w, rank, p_l):
    """
    calculate the error between f(x) and the real y
    :param x: the x coordinate of the training sample, which is a row vector [x1, x2, ..., xn]
    :param y: the y coordinate of the training sample, which is a row vector [y1, y2, ..., yn]
    :param w: the parameter omega, which is a row vector [w0, w1, ..., wn]
    :param rank: the rank of the polynomial function
    :param p_l: parameter in the regular term
    :return: the error
    """
    regular_term = 0.5 * p_l * np.sqrt(np.sum(w ** 2))
    error = np.sum((calc_fx(x, w, rank) - y) ** 2) * 0.5 + regular_term
    return error
