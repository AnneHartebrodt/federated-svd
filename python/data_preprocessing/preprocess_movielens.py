import numpy as np
import h5py
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as lsa
data = pd.read_csv('/home/anne/Downloads/ml-100k/u.data', header=None, sep='\t')
from scipy.sparse import dok_matrix
sp = sps.csc_matrix((data.iloc[:,3], (data.iloc[:,0], data.iloc[:, 1])), dtype='float32')

sp.tobsr(blocksize=(), copy=True)
u1,s1,v1 = lsa.svds(sp)
u1 = np.flip(u1)
v1 = np.flip(v1.T)

means = sp.mean(axis=0)
sp = sp.todense()
std = np.std(sp, axis=0)
std[std==0] = 1
sp =sp/std
u, s, v = la.svd(sp.T.dot(sp))

np.sqrt(s1)

pd.DataFrame(S).to_csv('/home/anne/Documents/featurecloud/singular-value-decomposition/data/tabular/movielens.tsv', sep='\t', header=False, index_label=None)

# data.dtype = 'float'

