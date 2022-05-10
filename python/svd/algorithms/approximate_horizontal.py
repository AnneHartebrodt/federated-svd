import numpy as np
from svd.python.svd.logging import *
import svd.python.svd.shared_functions as sh

def approximate_vertical(data_list, k=10, factor_k=2, filename=None):
    data_list = [d.T for d in data_list]
    v, e = simulate_federated_horizontal_pca(data_list, k=k, factor_k=factor_k, filename=filename)
    g = [np.dot(d, v)[:, 0:k] for d in data_list]
    return g, v


def simulate_federated_horizontal_pca(datasets, k=10, factor_k=2, filename=None):
    partial = []
    tol = TransmissionLogger()
    tol.open(filename)
    i = 0
    for d in datasets:
        #d = np.float32(d)
        dl = local_SVD(d, k=factor_k*k)
        partial.append(dl)
        tol.log_transmission( "H_local=CS", -1, i, dl)
        i =i+1
    print('Intermediate dimensions' +str(factor_k*k))
    dpca = aggregate_partial_SVDs(partial, factor_k*k)
    tol.log_transmission( "H_global=SC", -2, 1, dpca)
    tol.close()
    return dpca

def local_SVD(data, k=20):
    """
    Performs a singular value decomposition local data
    :param cov: A covariance matrix
    :param r: The number of top principal components to be considered
    :return: U_r*S_r (The product of the matrices taking the top r colums/rows)
    """

    # returns column vectors only
    U, S, UT, nd = sh.svd_sub(data, k)
    # In case we want to use more values in the approximation
    nd = min(nd, k)
    R = np.zeros((nd, nd))
    np.fill_diagonal(R, S[0:nd])
    U_r = UT[:, 0:nd]
    P = np.dot(np.sqrt(R), U_r.T)
    print(P.shape)
    return P

def aggregate_partial_SVDs(svds, t2=10):
    """
    Function assumes equally shaped covariances matrices.
    :param svd_list: List of local P matrices
    :return:
    """

    svds = np.concatenate(svds, axis=0)
    ndim = min(t2, svds.shape[0] - 1)

    print(svds.shape)
    U, S, UT, nd = sh.svd_sub(svds, ndim)
    return UT, S