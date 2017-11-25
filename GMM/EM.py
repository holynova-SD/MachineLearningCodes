import math


def e_step(sigma, gamma, mu, k, n, data_in):
    """
    the e_step in EM algorithm
    :param sigma: # the variance of these two Gaussian distributions.
    :param gamma: Expectation matrix.
    :param mu: the mean value of these two Gaussian distributions.
    :param k: number of different Gaussian distributions
    :param n: number of data
    :param data_in: original data
    :return: Expectation matrix after numbers of iteration.
    """
    for i in range(0, n):
        x = 0
        for j in range(0, k):
            x += math.exp((-1/(2*(float(sigma**2))))*(float(data_in[0, i]-mu[j]))**2)
        for j in range(0, k):
            y = math.exp((-1/(2*(float(sigma**2))))*(float(data_in[0, i]-mu[j]))**2)
            gamma[i, j] = y / x
    return gamma


def m_step(gamma, mu, k, n, data_in):
    """
    the m_step in EM algorithm
    :param gamma: Expectation matrix.
    :param mu: the mean value of these two Gaussian distributions.
    :param k: number of different Gaussian distributions
    :param n: number of data
    :param data_in: original data
    :return: the mean value of these two Gaussian distributions after numbers of iteration.
    """
    for j in range(0, k):
        x = y = 0
        for i in range(0, n):
            x += gamma[i, j] * data_in[0, i]
            y += gamma[i, j]
        mu[j] = x / y
    return mu
