from PCA import *
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    # data part
    n = 100
    mean = [0, 0, 0]
    cov = [[2, 3, 4], [1, 1, 1], [4, 3, 2]]
    # generate data
    x, y, z = np.random.multivariate_normal(mean, cov, n).T
    Data_in = np.mat([np.array([x[i], y[i], z[i]]) for i in range(len(x))])
    # calculate new data
    PCA_Data_in = np.mat(pca(Data_in, 2))
    # draw pictures
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='g')
    ax.scatter([PCA_Data_in[:, 0]], [PCA_Data_in[:, 1]], [PCA_Data_in[:, 2]], c='r')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

    # picture part
    picture_path = os.getcwd() + '/image/'
    output_path = os.getcwd() + '/output/'
    X = []
    n = 0
    # get all the pictures
    for name in glob.glob(picture_path + '*.pgm'):
        n += 1
        img = Image.open(name)
        X.append(np.array(img).reshape((1, 3840))[0])
    Data = np.mat(X)
    # calculate new data
    PCA_Data = pca(Data, 100)
    # calculate the signal to noise ratio
    all_diff = []
    for i in range(0, n):
        diff = np.abs(PCA_Data[i] - Data[i])
        MSE = np.sqrt(1/3840.0 * (np.array(diff)**2).sum())
        PSNR = 20 * np.log10(255/MSE)
        all_diff.append(PSNR)
    print(all_diff)
    # output the pictures
    for i in range(0, n):
        output = PCA_Data[i].copy()
        output = output.reshape(60, 64)
        output_pic = Image.fromarray(output.astype(np.uint8))
        output_pic.save(output_path + "result" + str(i) + ".pgm")
