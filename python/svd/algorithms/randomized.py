from svd.python.svd.algorithms.subspace_iteration import simulate_subspace_iteration
import scipy.linalg as la
import scipy.sparse.linalg as lsa
from svd.python.svd.logging import *
import svd.python.svd.shared_functions as sh

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
                                                                               previous_iterations=0)
    mot.start()

    H_stack = np.asarray(np.concatenate(H_stack, axis=1))
    H, S, G = lsa.svds(H_stack, k=H_stack.shape[1]-1)
    tol.log_transmission("H_global=SC", iterations+1, 1, H)
    tol.close()
    H = np.flip(H, axis=1)
    p = [np.dot(H.T,d) for d in data_list]
    covs = [np.dot(p1, p1.T) for p1 in p]
    u1,s1, v1 = lsa.svds(np.sum(covs, axis=0), k=k)
    u1 = np.flip(u1, axis=1)
    g1 = [np.dot(p1.T, u1) for p1 in p]
    G_i = np.concatenate(g1, axis=0)
    G_i, R = la.qr(G_i, mode='economic')
    mot.stop()
    log_time_keywords(filename, 'matrix_operations-randomized', mot.total())

    aol = AccuracyLogger()
    aol.open(filename)
    aol.log_current_accuracy(u=u, G_i=G_i, eigenvals=eigenvals, conv=None, current_iteration=iterations+1,
                             choices=choices, precomputed_pca=precomputed_pca, v=v, H_i=H_i)
    aol.close()
    return G_i


