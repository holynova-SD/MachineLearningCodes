# -*- coding: utf-8 -*-
import numpy as np


def ls(x, y, rank):
    """
    use least square method to calculate the parameter omega in the polynomial function
    :param x: the x coordinate of the training sample, which is a row vector [x1, x2, ..., xn]
    :param y: the y coordinate of the training sample, which is a row vector [y1, y2, ..., yn]
    :param rank: he rank of the polynomial function
    :return: omega we get using least square method
    """
    # generate a transpose of vandermonde matrix
    tran_van_matrix = np.tile(x, (rank + 1, 1))
    for r in range(rank + 1):
        tran_van_matrix[r] = tran_van_matrix[r] ** r

    reverse = np.linalg.inv(tran_van_matrix.dot(tran_van_matrix.transpose()))
    ls_omega = y.dot(tran_van_matrix.transpose()).dot(reverse)

    return ls_omega
