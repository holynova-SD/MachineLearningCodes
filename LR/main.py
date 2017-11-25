from Logistic_Regression import *
from Newton import *
import matplotlib.pyplot as plt


def generate_data():
    """
    generate the training data
    :return: data in standard format
    """
    train_x = []
    train_y = []
    file_in = open('testSet.txt')
    for line in file_in.readlines():
        line_array = line.strip().split()
        train_x.append([1.0, float(line_array[0]), float(line_array[1])])
        train_y.append(float(line_array[2]))
    return np.mat(train_x), np.mat(train_y).transpose()


if __name__ == '__main__':
    data_x, data_y = generate_data()

    omega = np.ones((3, 1))

    alpha = 0.01

    p_lambda = 0
    omega_gd = logistic_regression(data_x, data_y, omega, alpha, p_lambda)
    omega_nt = newton(data_x, data_y, omega)

    for i in range(100):
        if int(data_y[i, 0]) == 0:
            plt.plot(data_x[i, 1], data_x[i, 2], 'or')
        elif int(data_y[i, 0]) == 1:
            plt.plot(data_x[i, 1], data_x[i, 2], 'ob')

    min_x = min(data_x[:, 1])[0, 0]
    max_x = max(data_x[:, 1])[0, 0]

    omega_gd = omega_gd.getA()
    y_min_x_gd = float(-omega_gd[0] - omega_gd[1] * min_x) / omega_gd[2]
    y_max_x_gd = float(-omega_gd[0] - omega_gd[1] * max_x) / omega_gd[2]

    omega_nt = omega_nt.getA()
    y_min_x_nt = float(-omega_nt[0] - omega_nt[1] * min_x) / omega_nt[2]
    y_max_x_nt = float(-omega_nt[0] - omega_nt[1] * max_x) / omega_nt[2]

    plt.plot([min_x, max_x], [y_min_x_gd, y_max_x_gd], color="green")
    plt.plot([min_x, max_x], [y_min_x_nt, y_max_x_nt], color="black")
    plt.show()
