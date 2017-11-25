import numpy as np


def pca(data, k):
    """
    the PCA method
    :param data: input data set.
    :param k: the dimension you want to fall to.
    :return: new data set in the original vector space.
    """
    mean = np.mean(data, axis=0)
    data_after = data - mean
    covariance_matrix = np.cov(data_after, rowvar=False)

    eigenvalue, eigenvector = np.linalg.eig(covariance_matrix)

    eigenvalue_sort = np.argsort(eigenvalue)
    eigenvalue_sort = eigenvalue_sort[: -(k + 1): -1]
    eigenvector_after = eigenvector[:, eigenvalue_sort]

    pca_data = data_after.dot(eigenvector_after)

    pca_data = pca_data.dot(eigenvector_after.transpose()) + mean

    return pca_data
