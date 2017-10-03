# -*- coding: utf-8 -*-
import numpy as np

# set the random number seed
np.random.seed(234)


def init_data(n, start, end, sigma):
    """
    get a set of training sample
    :param n: number of samples
    :param start: the left end of the interval
    :param end: the right end of the interval
    :param sigma: the standard deviation of normal distribution which applied on the y coordinates as noise
    :return: a row vector of x coordinates and a row vector of y coordinates
    """
    x = np.random.random(n)
    x *= float(end - start)
    x += start
    x = np.sort(x)

    y = np.sin(2 * np.pi * x) + np.random.normal(0, sigma, n)

    return x, y


def init_omega(rank):
    """
    get a initial value of parameter omega
    :param rank: the rank of the polynomial function
    :return: a random value
    """
    w = np.random.random(rank + 1)

    return w
