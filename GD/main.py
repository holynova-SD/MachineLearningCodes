# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from init_data import *
from batch_gradient_descent import *
from least_square import *
from conjugate_gradient_descent import *


if __name__ == '__main__':

    # set the number of training samples
    sample_num = 10

    # set the rank of the polynomial function
    m = 5

    # initialize the training set
    data_x, data_y = init_data(sample_num, 0, 1, 0.05)

    # initialize the parameter omega
    omega = init_omega(m)

    # create a set of x to draw curves
    picture_x = np.random.random(100)
    picture_x = np.sort(picture_x)

    # use different method to get different omega, and generate different curve
    omega_bgd = bgd(data_x, data_y, omega, m)
    picture_y_bgd = calc_fx(picture_x, omega_bgd, m)
    plt.plot(picture_x, picture_y_bgd, label="batch gradient descent", color="red")

    omega_ls = ls(data_x, data_y, m)
    picture_y_ls = calc_fx(picture_x, omega_ls, m)
    plt.plot(picture_x, picture_y_ls, label="least square", color="green")

    omega_cgd = cgd(data_x, data_y, omega, m)
    picture_y_cgd = calc_fx(picture_x, omega_cgd, m)
    plt.plot(picture_x, picture_y_cgd, label="conjugate gradient descent", color="blue")

    plt.scatter(data_x, data_y, color="black")

    plt.legend()
    plt.show()



