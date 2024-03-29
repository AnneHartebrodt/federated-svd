'''
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


'''

import numpy as np
import scipy as sc
import svd.shared_functions as sh
import scipy.linalg as la
import svd.comparison as co



def local1(data, G_i):
    return data.dot(G_i)

def local2(data, H_pooled, G_previous):
    return G_previous + data.T.dot(H_pooled)

def pooling(H_all):
    H_pooled = H_all[0]
    for h in H_all[1:len(H_all)]:
        H_pooled = H_pooled + h
    #H_pooled = H_pooled / len(H_all)
    H_pooled = H_pooled / np.linalg.norm(H_pooled)
    return H_pooled



def standalone(data, k=10):
    G_i = sh.generate_random_gaussian(data.shape[1], k) # phi1
    G_i, Q = la.qr(G_i, mode='economic')
    converged = False
    previous = G_i
    previous_h = sh.generate_random_gaussian(data.shape[0], k)
    iterations = 0
    while not converged:
        iterations = iterations+1
        print(iterations)
        H_i = data.dot(G_i) # YiPhii , gamma if standalone
        G_i =  data.T.dot(H_i)  + previous
        G_i, Q = la.qr(G_i, mode='economic') # 'compute residuals'
        converged, sum = convergence_checker(H_i, previous_h, epsilon=0.0000001)
        previous_h = H_i
        previous = G_i
    return G_i

def convergence_checker_d(current, previous, epsilon=0.000001):
    '''
    Checks the convergence
    Args:
        H_i:
        N_list:
        previous:
        epsilon:

    Returns: -1 if convergence not reached

    '''
    sum = 0
    for i in range(current.shape[1]):
        ra = current[:, i].T.dot(current[:, i]) / sc.linalg.norm(current[:, i])
        sum = np.abs(ra)
    if np.abs(sum-previous) < epsilon:
        return -1
    else:
        return ra

def convergence_checker_angle(current, previous, epsilon=0.000001, return_converged=False):
    """
    Check if angle between consecutive eigenvectors is nearing 0
    Args:
        current: Current eigenvalue guesss
        previous: Previous eigenvalue guess
        epsilon: tolerance
        return_converged: verbosity

    Returns:

    """
    sum = 0
    converged = True
    converged_eigenvectors = []
    deltas = []
    for i in range(current.shape[1]):
        ra = current[:,i].T.dot(previous[:,i])
        if ra >= 1-epsilon:
            converged_eigenvectors.append(i)
        else:
            # if one of the eigenvalues has not converged set overall convergence to true
            converged = False
    if return_converged:
        return converged, sum,  converged_eigenvectors, deltas
    else:
        return converged, sum

def convergence_checker(current, previous, epsilon=0.000001, return_converged=False):
    """
    Convergence checked via Raleigh coefficient.
    Args:
        current:
        previous:
        epsilon:
        return_converged:

    Returns:

    """
    sum = 0
    converged = True
    converged_eigenvectors = []
    deltas = []
    for i in range(current.shape[1]):
        ra = current[:,i].T.dot(current[:,i])/sc.linalg.norm(current[:,i])
        rap = previous[:,i].T.dot(previous[:,i])/sc.linalg.norm(previous[:,i])
        sum = sum + np.abs(ra-rap)
        deltas.append(np.abs(ra-rap))
        if np.abs(ra-rap) > epsilon:
            converged=False
        else:
            converged_eigenvectors.append(i)
    if return_converged:
        return converged, sum,  converged_eigenvectors, deltas
    else:
        return converged, sum

def convergence_checker_a(current, previous,alpha_norm, alpha_norm_prev, epsilon=0.000001, return_converged=False):
    """
    weird convergence criterion emplyed by Guo et al.
    Args:
        current: Current H
        previous: previous H
        alpha_norm: Current G norm
        alpha_norm_prev: previous G norm
        epsilon: tolerance
        return_converged: Which values to return

    Returns:

    """
    sum = 0
    converged = True
    converged_eigenvectors = []
    deltas = []
    for i in range(current.shape[1]):
        ra = current[:,i].T.dot(current[:,i])/alpha_norm[i]
        rap = previous[:,i].T.dot(previous[:,i])/alpha_norm_prev[i]
        sum = sum + np.abs(ra-rap)
        deltas.append(np.abs(ra-rap))
        if np.abs(ra-rap) > epsilon:
            converged=False
        else:
            converged_eigenvectors.append(i)
    if return_converged:
        return converged, sum,  converged_eigenvectors, deltas
    else:
        return converged, sum

def standalone2(data, first):

    G_i = sh.generate_random_gaussian(data.shape[1], 1)
    G_i = G_i - np.inner(G_i, first.T).dot(first.T)
    print(np.asarray(G_i).T)
    print(first)
    print(co.angle(np.asarray(G_i).T, np.asarray(first).T))
    Q, R = la.qr(np.asarray(np.concatenate([ first.T,G_i], axis = 1)))
    print(Q[:,0])
    G_i = Q[:,1]
    converged = False
    previous = G_i
    previous_h = sh.generate_random_gaussian(data.shape[0], 1)
    iterations = 0
    while not converged:
        iterations = iterations+1
        print(iterations)
        H_i = data.dot(G_i) # YiPhii , gamma if standalone
        G_i =  data.T.dot(H_i)  + previous
        Q, R = la.qr(np.asarray(np.stack([first.flatten(),G_i.T])).T)
        G_i = Q[:, 1]
        converged = convergence_checker(H_i, previous_h)
        previous_h = H_i
        previous = G_i
    return G_i


def get_initial_eigenvector_k(V):
    '''

     ap = a -sum over k-1 <a, ai>ai
    Args:
        V:

    Returns:


    '''

    a = sh.generate_random_gaussian(1, V.shape[0])
    sum = np.zeros(V.shape[0])
    for v in range(V.shape[1]):
        sum = sum + a.dot(v).dot(v)
    ap = a - sum
    return ap