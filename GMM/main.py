from EM import *
import numpy as np


# Suppose here are two Gaussian distribution, and the variance is known.

def init_data(sigma, mu_1, mu_2, n):
    """
    Generate data that is a mixture of two Gaussian distribution.
    :param sigma: the variance of the  Gaussian distribution. Suppose it is same for those two distribution.
    :param mu_1: the mean value of one Gaussian distribution.
    :param mu_2: the mean value of another Gaussian distribution.
    :param n: number of data.
    :return: the data set and the initialized Expectation matrix.
    """
    data = np.zeros((1, n))
    gamma = np.zeros((n, 2))
    for j in range(0, n):
        if np.random.random(1) > 0.5:
            data[0, j] = np.random.normal() * sigma + mu_1
        else:
            data[0, j] = np.random.normal() * sigma + mu_2
    return data, gamma


if __name__ == '__main__':
    # the variance of these two Gaussian distribution
    Sigma = 5
    # the mean value of these Gaussian distribution
    Mu = np.zeros(2) + [20, 30]
    # number of different Gaussian distributions
    K = 2
    # number of data
    N = 1000
    # allowable error
    epsilon = 0.00001

    Data, Gamma = init_data(Sigma, Mu[0], Mu[1], N)

    Mu_test = np.random.random(2)
    # the EM algorithm
    for i in range(0, N):
        Mu_test_before = Mu_test.copy()
        Gamma = e_step(Sigma, Gamma, Mu_test, K, N, Data)
        Mu_test = m_step(Gamma, Mu_test, K, N, Data)
        print(Mu_test)
        if sum(abs(Mu_test - Mu_test_before)) < epsilon:
            break


