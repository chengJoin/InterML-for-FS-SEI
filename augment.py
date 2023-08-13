import numpy as np


def rotate_matrix(theta):
    m = np.zeros((2, 2))
    m[0, 0] = np.cos(theta)
    m[0, 1] = -np.sin(theta)
    m[1, 0] = np.sin(theta)
    m[1, 1] = np.cos(theta)
    return m


def Rotate_DA(x, y):
    [N, L, C] = np.shape(x)
    x_rotate1 = np.matmul(x, rotate_matrix(np.pi / 2))
    x_rotate2 = np.matmul(x, rotate_matrix(np.pi))
    x_rotate3 = np.matmul(x, rotate_matrix(3 * np.pi / 2))

    x_DA = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))

    y_DA = np.tile(y, (1, 4))
    y_DA = y_DA.T
    y_DA = y_DA.reshape(-1)
    y_DA = y_DA.T
    return x_DA, y_DA
