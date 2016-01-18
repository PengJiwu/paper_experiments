__author__ = 'tao'
#-*- coding:utf-8 -*-

import numpy as np

def isvd(X, maxiterations = 20, tolerance = 0.001, display = False):
    X[X == 0] = np.nan

    if not True in np.isnan(X):
        U, s, V = np.linalg.svd(X, False)
        S = np.diag(s)
        return U, S, V

    D, N = X.shape
    M = min(D, N)
    G = np.zeros((D, N))
    G[np.isnan(X)] = 1

    XX = np.copy(X)
    XX[np.isnan(XX)] = np.nanmean(XX)
    B, s, V = np.linalg.svd(XX, False)

    W = np.empty((M, N))
    old_msqerror = np.nan
    for loop in xrange(maxiterations):
        for n in xrange(N):
            comprow = G[:, n] == 0
            Mn = B[comprow, :].transpose().dot(B[comprow, :])
            cn = B[comprow, :].transpose().dot(X[comprow, n])
            W[:, n] = np.linalg.pinv(Mn).dot(cn)

        for d in range(D):
            goodn = G[d, :] == 0
            Wg = W[:, goodn]
            Fd = Wg.dot(Wg.transpose())
            md = W[:, goodn].dot(X[d, goodn].transpose())
            B[d, :] = np.linalg.pinv(Fd).dot(md).transpose()

        msqerror = np.nanmean((X - B.dot(W))**2)
        if display:
            print 'Iteration %d: mean square reconstruction of non missing elements = %g\n' % (loop, msqerror)
        if abs(msqerror - old_msqerror) < tolerance:
            break
        else:
            old_msqerror = msqerror

    U, s, V = np.linalg.svd(B.dot(W), False)
    S = np.diag(s)
    return U, S, V