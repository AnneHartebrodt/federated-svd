import scipy.linalg as la
import svd.shared_functions as sh
import svd.algorithms as qr
from svd.logging import *
from svd.logging import TimerCounter

####### ORIGINAL POWER ITERATION SCHEME #######
def residuals(V, a=None, sums=None):
    """
    Compute the residuals according to the formula
    """
    if a is None:
        a = sh.generate_random_gaussian(1, V.shape[0])
    sum = np.zeros(V.shape[0])
    if sums is not None:
        for v in range(V.shape[1]):
            sum = sum + sums[v] * V[:, v].T
    else:
        for v in range(V.shape[1]):
            sum = sum + a.dot(V[:, v:v + 1]) * V[:, v].T
    ap = a - sum
    return ap


def simulate_guo(local_data, maxit, V_k=None, starting_vector=None, filename=None, u=None, choices=None,
                 precomputed_pca=None, federated_qr=False, v=None, gradient=False, epsilon=1e-9, guo_epsilon=1e-11,
                 log=True, previous_iterations=None):
    """
    Retrieve the first eigenvector
    Args:
        local_data: List of numpy arrays containing the data. The data has to be scaled already.
        k: The number of countermensions to retrieve
        maxit: Maximal number of iterations

    Returns: A column vector array containing the global eigenvectors

    """
    tol = TransmissionLogger()
    tol.open(filename)
    aol = AccuracyLogger()
    aol.open(filename)
    if previous_iterations is None:
        iterations = 0  # iteration counter
    else:
        iterations = previous_iterations

    # allow maxit for very eigenvector
    convergedH = False
    total_len = 0  # total number of samples/individuals in data
    for d in local_data:
        total_len = total_len + d.shape[1]
    if V_k is not None:
        id = V_k.shape[1]
    else:
        id = 0
    if starting_vector is None:
        # if there is no starting vector we generate an orthogonal
        # vector and start iterating
        if V_k is not None:
            # if it is not the first eigenvector use residuals
            # to orthogonalise
            G_i = residuals(V_k).T
        else:
            # if it is the first eigenvecto just generate
            # randonly and scale to unit norm
            G_i = sh.generate_random_gaussian(total_len, 1)
            G_i = G_i / np.linalg.norm(G_i)  # normalize
    else:
        # the vector is already preiterated and is orthonormal, we
        # just have to assure it is a 2d array.
        G_i = np.reshape(starting_vector, (total_len, 1))

    start = 0  # running variable to partition G_i
    G_list = []  # this are the parital eigevenctors
    Vk_list = []  # these are the partial eigenvectors already iterated
    for i in range(len(local_data)):
        G_list.append(G_i[start:start + local_data[i].shape[1], :])
        tol.log_transmission("G_i=SC", iterations, i, G_list[i], id+1)
        if V_k is not None:
            Vk_list.append(V_k[start:start + local_data[i].shape[1], :])
        start = start + local_data[i].shape[1]
    H_i_prev = sh.generate_random_gaussian(local_data[0].shape[0], 1)
    G_i_prev = G_i
    gi_norm_prev = la.norm(H_i_prev)
    mot = TimerCounter()
    while not convergedH and iterations < maxit:
        iterations = iterations + 1
        H_i = np.zeros((local_data[0].shape[0], 1))  # dummy initialise the H_i matrix
        mot.start()
        for i in range(len(local_data)):
            H_local = local_data[i].dot(G_list[i])
            tol.log_transmission( "H_local=CS", iterations, i, H_local ,id+1)
            H_i = H_i + H_local
        tol.log_transmission( "H_global=SC", iterations, 1, H_i ,id+1)

        for i in range(len(local_data)):
            if gradient:
                G_list[i] = local_data[i].T.dot( H_i) + G_list[i]
            else:
                G_list[i] = local_data[i].T.dot( H_i)
        mot.stop()
        gi_norm = 0
        if V_k is None:
            # compute the norm of the eigenvector and done.
            for i in range(len(G_list)):
                local_norm = np.sum(np.square(G_list[i]))
                gi_norm = gi_norm + np.sum(np.square(G_list[i]))
                tol.log_transmission( "local_norm=CS", iterations, i, local_norm ,id+1)

        elif federated_qr:
            temp = []
            for i in range(len(G_list)):
                gi_norm = gi_norm + np.sum(np.square(G_list[i]))
                temp.append(np.concatenate([Vk_list[i], G_list[i]], axis=1))

            G_i, G_list = qr.simulate_federated_qr(temp, encrypt=False)

            for i in range(len(G_list)):
                G_list[i] = G_list[i][:, id:id + 1]

        else:
            local_sums = []
            for i in range(len(G_list)):
                sum = []
                for vi in range(Vk_list[i].shape[1]):
                    dp = G_list[i].T.dot(Vk_list[i][:, vi:vi + 1]).flatten()
                    sum.append(dp)
                # cast to numpy array to determine size
                tol.log_transmission( "local_dot_prod=CS", iterations, i, np.asarray(sum), id+1)
                local_sums.append(sum)
            local_sums = np.asarray(local_sums)
            local_sums = np.sum(local_sums, axis=0).flatten()  # flatten to remove nesting
            tol.log_transmission( "global_dot_prod=SC", iterations, 1, local_sums, id+1)

            for i in range(len(G_list)):
                ap = G_list[i]
                for vi in range(Vk_list[i].shape[1]):
                    it = local_sums[vi] * Vk_list[i][:, vi:vi + 1].T
                    it = np.reshape(it, ap.shape)
                    ap = ap - it
                G_list[i] = ap

            for i in range(len(G_list)):
                c_n = np.sum(np.square(G_list[i]))
                tol.log_transmission("local_norm=CS", iterations, i, c_n, id+1)
                gi_norm = gi_norm + c_n

        gi_norm = np.sqrt(gi_norm)
        if V_k is None or not federated_qr:
            for i in range(len(G_list)):
                G_list[i] = G_list[i] / gi_norm
                if log:
                    # subsequent orthonormalisation at agrgegator
                    # (only current G_i because rest is assumed to be stored at agrgegator)
                    tol.log_transmission( "ALT_G_i_local=CS", iterations, i, G_list[i], id+1)
                    tol.log_transmission( "ALT_G_i=SC", iterations, i, G_list[i], id+1)


        #if gradient:
        convergedH, deltaH = sh.convergence_checker_rayleigh(H_i, H_i_prev, [gi_norm], [gi_norm_prev] ,epsilon=guo_epsilon)
        #else:
        #    convergedH, deltaH = sh.eigenvector_convergence_checker(H_i, H_i_prev, tolerance=epsilon)
        #    print(convergedH)
        #    print(deltaH)
        H_i_prev = H_i
        gi_norm_prev = gi_norm
        tol.log_transmission( "global_norm=SC", iterations, 1, gi_norm, id+1)

        G_i = np.concatenate(G_list, axis=0)
        G_i = np.asarray(G_i)
        H_i = np.asarray(H_i)
        aol.log_current_accuracy(u[:, id:id + 1], G_i, eigenvals=[gi_norm], conv=deltaH, current_iteration=iterations,
                             choices=choices, precomputed_pca=precomputed_pca, current_ev=id + 1,
                                v=v[:, id:id + 1], H_i=H_i)
    # return a complete eigenvector
    #print(gi_norm)
    print(iterations)
    log_time_keywords(filename, 'matrix_operations-subspace_iteration', mot.total())
    tol.close()
    aol.close()
    return G_i, iterations


def compute_k_eigenvectors(data_list, k, maxit, filename=None, u=None, choices=None, precomputed_pca=None,
                           federated_qr=False, v=None, gradient=True, epsilon=1e-9, guo_epsilon=1e-11):
    ug, it = simulate_guo(data_list, maxit=maxit, filename=filename, u=u, choices=choices, federated_qr=federated_qr, v=v,
                      gradient=gradient, epsilon=epsilon, guo_epsilon=guo_epsilon)
    u_all = ug
    for i in range(1, k):
        ug2, it= simulate_guo(data_list, maxit=maxit+it, V_k=u_all, filename=filename, u=u, choices=choices,
                           federated_qr=federated_qr, v=v, gradient=gradient, epsilon=epsilon, guo_epsilon=guo_epsilon,
                              previous_iterations=it)
        u_all = np.concatenate([u_all, ug2], axis=1)
    return u_all

