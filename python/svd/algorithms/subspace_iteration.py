"""
    Copyright (C) 2020 Anne Hartebrodt

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    Authors: Anne Hartebrodt

"""
import numpy as np
import scipy.linalg as la
import svd.shared_functions as sh
import svd.algorithms.qr as qr
from svd.logging import *


####### MATRIX POWER ITERATION SCHEME #######
def simulate_subspace_iteration(local_data, k, maxit, filename=None, u=None, choices=None, precomputed_pca=None, fractev=1.0,
                           federated_qr=False, v=None, gradient=False, epsilon=10e-9, g_ortho_freq=1000, g_init = None,
                                previous_iterations=None):
    """
    Simulate a federated run of principal component analysis using Guo et als algorithm in a modified version.

    Args:
        local_data: List of numpy arrays containing the data. The data has to be scaled already.
        k: The number of dimensions to retrieve
        maxit: Maximal number of iterations

    Returns: A column vector array containing the global eigenvectors

    """

    print(filename)
    G_list = []


    convergedH = False
    total_len = 0
    # generate an intitial  orthogonal noise matrix
    for d in local_data:
        total_len = total_len + d.shape[1]
        print(d.shape)
    start = 0

    if g_init is None:
        G_i = sh.generate_random_gaussian(total_len, k)
        G_i, R = la.qr(G_i, mode='economic')
        iterations = 0
    else:
        G_i = np.concatenate(g_init, axis=0)
        iterations = 1

    if previous_iterations is not None:
        iterations = previous_iterations

    tol = TransmissionLogger()
    tol.open(filename)

    aol = AccuracyLogger()
    aol.open(filename)

    # send parts to local sites
    for i in range(len(local_data)):
        G_list.append(G_i[start:start + local_data[i].shape[1], :])
        tol.log_transmission("G_i=SC", iterations, i, G_list[i])
        start = start + local_data[i].shape[1]

    # Initial guess
    H_i_prev = sh.generate_random_gaussian(local_data[0].shape[0], k)
    G_i_prev = G_i
    converged_eigenvals = []
    H_stack = []
    eigenvals_prev = None
    # Convergence can be reached when eigenvectors have converged, the maximal number of
    # iterations is reached or a predetermined number of eignevectors have converged.

    mot = TimerCounter()

    while not convergedH and iterations < maxit and len(converged_eigenvals) < k * fractev:
        iterations = iterations + 1
        print(iterations)
        # add up the H matrices
        H_i = np.zeros((local_data[0].shape[0], k))
        for i in range(len(local_data)):
            # send local H matrices to server
            mot.start()
            H_local = np.dot(local_data[i], G_list[i])
            mot.stop()
            tol.log_transmission("H_local=CS", iterations, i, H_local)
            # add up H matrices at server and send them back to the clients
            H_i = H_i + H_local

        # Log only once for one site
        tol.log_transmission( "H_global=SC", iterations, 1, H_i)

        # free orthonormalisation in terms of transmission cost
        mot.start()
        H_i, R = la.qr(H_i, mode='economic')
        mot.stop()

        # Eigenvector update
        for i in range(len(G_list)):
            # Use gradient based update of the Eigenvectors

            if gradient:
                G_list[i] = np.dot(local_data[i].T, H_i) + G_list[i]
            else:
                # Use power iterations based update of the eigenvalue scheme
                mot.start()
                G_list[i] = np.dot(local_data[i].T, H_i)
                mot.stop()
                if not federated_qr:
                    tol.log_transmission("Gi_local=CS", iterations, i, G_list[i])

        # This is just for logging purposes
        G_i = np.concatenate(G_list, axis=0)

        # Eigenvalues are the norms of the eigenvecotrs
        eigenvals = []
        for col in range(G_i.shape[1]):
            eigenvals.append(np.linalg.norm(G_i[:, col]))

        # this is not timed because it is done for logging purpose and not algorithmic reasons
        G_i, R = la.qr(G_i, mode='economic')


        convergedH, deltaH = sh.eigenvector_convergence_checker(H_i, H_i_prev, tolerance=epsilon)
        # use guos convergence criterion for comparison
        #convergedH, deltaH = sh.convergence_checker_rayleigh(H_i, H_i_prev, eigenvals, eigenvals_prev, epsilon=1e-11)
        # just out of curiousity, log the
        #convergedG, deltaG = sh.eigenvector_convergence_checker(G_i, G_i_prev, tolerance=epsilon)
        H_i_prev = H_i
        G_i_prev = G_i
        if iterations < 10:
            H_stack.append(H_i)

        aol.log_current_accuracy(u=u, G_i=G_i, eigenvals=eigenvals, conv=deltaH, current_iteration=iterations,
                                 choices=choices, precomputed_pca=precomputed_pca, v=v, H_i=H_i)
    # log the time for matrix operations
    log_time_keywords(filename, 'matrix_operations-subspace_iteration', mot.total())
    tol.close()
    aol.close()
    ortho, G_list, r, rlist = qr.simulate_federated_qr(G_list)
    return G_i, eigenvals, converged_eigenvals, H_i, H_stack, iterations, G_list



