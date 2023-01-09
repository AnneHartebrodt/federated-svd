import pandas as pd
import scipy.sparse as sps
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as lsa
import scipy as sc
from svd.comparison import compute_angles
import svd.shared_functions as sh
data = pd.read_csv('/home/anne/Downloads/ml-100k/u.data', header=None, sep='\t')
data = sps.csc_matrix((data.iloc[:, 3], (data.iloc[:, 0], data.iloc[:, 1])), dtype='float32')
print(data.shape)
# data = data.todense()
print(data.shape)
scale = True
center = True
if scale or center:
    means = data.mean(axis=0)
    data = data.todense()
    std = np.std(data, axis=0)
    if center:
        data = np.subtract(data, means)

    # self.tabdata.scaled = np.delete(self.tabdata.scaled, remove)
    # self.tabdata.rows = np.delete(self.tabdata.rows, remove)
    if scale:
        data = data / std

    if center:
        # impute. After centering, the mean should be 0, so this effectively mean imputation
        data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)

dataset_name = 'movielens'
data_list, choice  = sh.partition_data_horizontally(data, 2, randomize=False)

k = 40
u,s,v = lsa.svds(data, k=k)
u = np.flip(u, axis=1)
v = np.flip(v.T, axis=1)


for b in range(10):
    H_stack = []

    for i in range(10):
        Hl = np.zeros((data.shape[1],2*k))
        for d in data_list:
            H = np.random.random((d.shape[1], 2*k))
            H, r = la.qr(H, mode='economic')
            G = data.dot(H)
            G, r = la.qr(G, mode='economic')
            H = data.T.dot(G)
            Hl = np.add(Hl, H)
        H, r = la.qr(H, mode='economic')
        H_stack.append(H)


    H_stack = np.asarray(np.concatenate(H_stack, axis=1))
    H, S, G = lsa.svds(H_stack, k=H_stack.shape[1]-1)
    H = np.flip(H, axis=1)
    pl = [H.T.dot(d.T) for d in data_list]

    cov = [p.dot(p.T) for p in pl]

    u1,s1, v1 = lsa.svds(np.sum(cov, axis=0), k=k)
    u1 = np.flip(u1, axis=1)
    g1 = [p.T.dot(u1) for p in pl]
    g1 = np.concatenate(g1, axis = 0)
    G_i, R = la.qr(g1, mode='economic')
    print(compute_angles(u,G_i))





for b in range(10):
    H_stack = []

    for i in range(10):
        H = np.random.random((data.shape[0], 2*k))
        H, r = la.qr(H, mode='economic')
        G = data.T.dot(H)
        G, r = la.qr(G, mode='economic')
        H = data.dot(G)
        H, r = la.qr(H, mode='economic')
        H_stack.append(H)


    H_stack = np.asarray(np.concatenate(H_stack, axis=1))
    H, S, G = lsa.svds(H_stack, k=H_stack.shape[1]-1)
    H = np.flip(H, axis=1)
    p = H.T.dot(data)

    cov = p.dot(p.T)
    u1,s1, v1 = lsa.svds(cov, k=k)
    u1 = np.flip(u1, axis=1)
    g1 = p.T.dot(u1)
    G_i, R = la.qr(g1, mode='economic')


    print(compute_angles(v,G_i))

