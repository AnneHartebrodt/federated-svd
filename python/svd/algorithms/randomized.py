from svd.algorithms.subspace_iteration import simulate_subspace_iteration
import scipy.linalg as la
import scipy.sparse.linalg as lsa
from svd.logging import *
import svd.shared_functions as sh
import sklearn.utils.extmath as ex
import scipy.sparse as sc

def run_randomized(data_list, k, I, factor_k=2,filename=None, u=None, choices=None, precomputed_pca=None, fractev=1.0,
                           federated_qr=False, v=None, gradient=False, epsilon=10e-9, g_ortho_freq=1, g_init = None):
    mot = TimerCounter()
    tol = TransmissionLogger()
    tol.open(filename)
    G_i, eigenvals, converged_eigenvals, H_i, H_stack, iterations, G_list = simulate_subspace_iteration(data_list,
                                                                              k=factor_k*k,
                                                                                maxit= I,
                                                                               filename=filename,
                                                                               u=u,
                                                                               choices=choices,
                                                                               precomputed_pca=precomputed_pca,
                                                                               fractev=fractev,
                                                                               federated_qr=federated_qr,
                                                                               v=v,
                                                                                gradient=gradient,
                                                                               epsilon=epsilon,
                                                                               g_ortho_freq=g_ortho_freq,
                                                                               g_init = g_init,
                                                                               previous_iterations=0, final_ortho=False)
    mot.start()

    print('project')
    h = [H.shape for H in H_stack]
    H_stack = np.asarray(np.concatenate(H_stack, axis=1))
    print(H_stack.shape)
    if H_stack.shape[0]*H_stack.shape[1]>H_stack.shape[1]*H_stack.shape[1]:
        temp_cov = H_stack.T.dot(H_stack)
        print(temp_cov.shape)
        # This is faster than scipy.sparse.linalg
        # This is also in the 'usual' orientation
        u_temp, s_temp, v_temp = la.svd(temp_cov)
        H_temp = H_stack.dot(u_temp)
        print(H_temp.shape)
        H, r = la.qr(H_temp, mode='economic')
    else:
        H, S, G = lsa.svds(H_stack, k=H_stack.shape[1]-1)
        H = np.flip(H, axis=1)
    tol.log_transmission("H_global=SC", iterations+1, 1, H)
    tol.close()
    print('aggregated')
    H = np.flip(H, axis=1)
    print('Project data')
    if sc.issparse(data_list[0]):
        print('Sparse matrix')
        p = [ex.safe_sparse_dot(H.T,d) for d in data_list]
    else:
        print('Dense matrix')
        p = [H.T.dot(d) for d in data_list]
    print('Compute covariance')
    covs = [p1.dot(p1.T) for p1 in p]
    print('compute inner SVD')
    u1,s1, v1 = lsa.svds(np.sum(covs, axis=0), k=k)
    u1 = np.flip(u1, axis=1)
    print('upscale')
    g1 = [p1.T.dot(u1) for p1 in p]
    print('finalize')
    G_i = np.concatenate(g1, axis=0)
    print('orthonormalize')
    G_i, R = la.qr(G_i, mode='economic')
    print('done')
    mot.stop()
    log_time_keywords(filename, 'matrix_operations-randomized', mot.total())

    aol = AccuracyLogger()
    aol.open(filename)
    aol.log_current_accuracy(u=u, G_i=G_i, eigenvals=eigenvals, conv=None, current_iteration=iterations+1,
                             choices=choices, precomputed_pca=precomputed_pca, v=v, H_i=H_i)
    aol.close()
    return G_i


