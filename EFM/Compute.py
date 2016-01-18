#-*- coding:utf-8 -*-

__author__ = 'tao'

import pandas as pd
import numpy as np
import math

from recommender import recommend_items

def _process_coefficient(matrix):
    for idx, row in matrix.iterrows():
        for cdx, val in row.iteritems():
            if val == 0 or math.isnan(val) or math.isinf(val):
                row[cdx] = 1


def _update_V(lamda_x, lamda_y, lamda_v, X, Y, U_1, U_2, V):
    numerator = lamda_x * X.T.dot(U_1) + lamda_y * Y.T.dot(U_2)
    r = U_1.shape[1]
    I = pd.DataFrame(np.ones((r, r)))
    denominator = V.dot(lamda_x * U_1.T.dot(U_1) + lamda_y * U_2.T.dot(U_2) + lamda_v * I)

    coefficient = numerator.div(denominator)
    coefficient = coefficient.applymap(math.sqrt)
    _process_coefficient(coefficient)

    #print coefficient.ix[:4,:4]

    new_V = V.mul(coefficient)
    return new_V

def _update_U_1(lamda_x, lamda_u, A, X, U_1, U_2, V, H_1, H_2):
    numerator = A.dot(U_2) + lamda_x * X.dot(V)
    r = U_1.shape[1]
    I = pd.DataFrame(np.ones((r, r)))
    denominator = (U_1.dot(U_2.T) + H_1.dot(H_2.T)).dot(U_2) + U_1.dot(lamda_x * V.T.dot(V) + lamda_u * I)

    coefficient = numerator.div(denominator)
    coefficient = coefficient.applymap(math.sqrt)
    _process_coefficient(coefficient)

    new_U_1 = U_1.mul(coefficient)
    return new_U_1

def _update_U_2(lamda_y, lamda_u, A, Y, U_1, U_2, V, H_1, H_2):
    numerator = A.T.dot(U_1) + lamda_y * Y.dot(V)
    r = U_1.shape[1]
    I = pd.DataFrame(np.ones((r, r)))
    denominator = (U_2.dot(U_1.T) + H_2.dot(H_1.T)).dot(U_1) + U_2.dot(lamda_y * V.T.dot(V) + lamda_u * I)

    coefficient = numerator.div(denominator)
    coefficient = coefficient.applymap(math.sqrt)
    _process_coefficient(coefficient)

    new_U_2 = U_2.mul(coefficient)
    return new_U_2

def _update_H_1(lamda_h, A, U_1, U_2, H_1, H_2):
    numerator = A.dot(H_2)
    denominator = (U_1.dot(U_2.T) + H_1.dot(H_2.T)).dot(H_2) + lamda_h * H_1

    coefficient = numerator.div(denominator)
    coefficient = coefficient.applymap(math.sqrt)
    _process_coefficient(coefficient)

    new_H_1 = H_1.mul(coefficient)
    return new_H_1


def _update_H_2(lamda_h, A, U_1, U_2, H_1, H_2):
    numerator = A.T.dot(H_1)
    denominator = (U_2.dot(U_1.T) + H_2.dot(H_1.T)).dot(H_1) + lamda_h * H_2

    coefficient = numerator.div(denominator)
    coefficient = coefficient.applymap(math.sqrt)
    _process_coefficient(coefficient)

    new_H_2 = H_2.mul(coefficient)
    return new_H_2

def _loss(X, Y, A, U_1, U_2, V, H_1, H_2, lamda_x, lamda_y, lamda_u, lamda_h, lamda_v):
    n1 = np.linalg.norm(((U_1.dot(U_2.T) + H_1.dot(H_2.T)) - A).as_matrix())
    n2 = lamda_x * np.linalg.norm((U_1.dot(V.T) - X).as_matrix())
    n3 = lamda_y * np.linalg.norm((U_2.dot(V.T) - Y).as_matrix())
    n4 = lamda_u * (np.linalg.norm(U_1.as_matrix()) + np.linalg.norm(U_2.as_matrix()))
    n5 = lamda_h * (np.linalg.norm(H_1.as_matrix()) + np.linalg.norm(H_2.as_matrix()))
    n6 = lamda_v * np.linalg.norm(V.as_matrix())

    return n1 + n2 + n3 + n4 + n5 + n6


def _is_convergent(X, Y, A, V, new_V, U_1, new_U_1, U_2, new_U_2, H_1, new_H_1, H_2, new_H_2, lamda_x, lamda_y, lamda_u, lamda_h, lamda_v, convergence):
    pre_loss = _loss(X, Y, A, U_1, U_2, V, H_1, H_2, lamda_x, lamda_y, lamda_u, lamda_h, lamda_v)
    now_loss = _loss(X, Y, A, new_U_1, new_U_2, new_V, new_H_1, new_H_2, lamda_x, lamda_y, lamda_u, lamda_h, lamda_v)
    #print 'now_loss: ', now_loss

    iter_rate = abs(now_loss - pre_loss) / pre_loss

    if iter_rate < convergence:
        return True
    else:
        return False


def compute(A, X, Y,  user_num, item_num, feature_num, r, r_p, lamda_x, lamda_y, lamda_u, lamda_h, lamda_v, convergence, T):
    U_1 = pd.DataFrame(np.random.rand(user_num, r))
    U_2 = pd.DataFrame(np.random.rand(item_num, r))
    V = pd.DataFrame(np.random.rand(feature_num, r))
    H_1 = pd.DataFrame(np.random.rand(user_num, r_p))
    H_2 = pd.DataFrame(np.random.rand(item_num, r_p))

    #recommend(U_1, U_2, V, H_1, H_2, set([]), 0.5, 50, 5)

    for t in range(T):
        new_V = _update_V(lamda_x, lamda_y, lamda_v, X, Y, U_1, U_2, V)
        new_U_1 = _update_U_1(lamda_x, lamda_u, A, X, U_1, U_2, V, H_1, H_2)
        new_U_2 = _update_U_2(lamda_y, lamda_u, A, Y, U_1, U_2, V, H_1, H_2)
        new_H_1 = _update_H_1(lamda_h, A, U_1, U_2, H_1, H_2)
        new_H_2 = _update_H_2(lamda_h, A, U_1, U_2, H_1, H_2)

        is_convergent = _is_convergent(X, Y, A, V, new_V, U_1, new_U_1, U_2, new_U_2, H_1, new_H_1, H_2, new_H_2, lamda_x, lamda_y, lamda_u, lamda_h, lamda_v, convergence)
        if is_convergent:
            print "Convergent: ", t
            break

        #print new_V.ix[:29,:5],new_U_1.ix[:29,:5],new_U_2.ix[:29,:5],new_H_1.ix[:29,:5],new_H_2.ix[:29,:5],
        V = new_V
        U_1 = new_U_1
        U_2 = new_U_2
        H_1 = new_H_1
        H_2 = new_H_2

        #recommend(U_1, U_2, V, H_1, H_2, set([]), 0.5, 50, 5)

    return U_1, U_2, V, H_1, H_2