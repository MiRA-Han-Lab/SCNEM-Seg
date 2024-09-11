from scipy.ndimage import affine_transform
import numpy as np
import math
import multiprocessing
from concurrent import futures

def compute_matrix(matrix_cell, i):
    if i == 1:
        matrix = matrix_cell[i - 1]
    else:
        matrix = np.linalg.multi_dot(matrix_cell[0:i])
    return np.linalg.inv(matrix)

def affine_trans_block(raw, matrix_cell, affine_scale, Whether_invert=False, order=1, n_threads=None):
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    # matrix_cell = matrix_cell[-raw.shape[0] + 1:]
    # matrix_cell = np.copy(matrix_cell)
    for i in range(raw.shape[0] - 1):
        matrix_cell[i][0, 2] = matrix_cell[i][0, 2] * affine_scale
        matrix_cell[i][1, 2] = matrix_cell[i][1, 2] * affine_scale
        error_index = matrix_cell[i][0, 0] * matrix_cell[i][1, 1] - matrix_cell[i][0, 1] * matrix_cell[i][1, 0]
        if error_index > 1.2 or error_index < 0.8:
            print('error, change %d' % i)
            matrix_cell[i] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # out = np.zeros_like(raw)
    def _affine_z(i):
    # for i in range(raw.shape[0]):
        im = raw[i].T
        if i == 0:
            out_z = im.T
        else:
            matrix = compute_matrix(matrix_cell, i)
            if Whether_invert == True:
                matrix = np.linalg.inv(matrix)
            out_z = affine_transform(im, matrix, order=order).T ########affine_transform #the parameter for matrix is YX and inversed
        return out_z

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_affine_z, z) for z in range(raw.shape[0])]
        out = [t.result() for t in tasks]

    return np.stack(out,axis=0)











