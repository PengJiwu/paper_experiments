__author__ = 'tao'

import pandas as pd
import numpy as np
from scipy.stats import norm

'''
import sklearn.mixture

gmm = sklearn.mixture.GMM()

def _gaussian_curve_fit(data):
    r = gmm.fit(data[:, np.newaxis])
    mu = r.means_[0,0]
    sigm = r.covars_[0,0]

    return mu, sigm
'''

mu_param_lambda = 1
sigm_param_lambda = 0.5

def _gaussian_curve_fit(single_data, category_averg_mu_and_sigma):
    category_mu, category_sigm = category_averg_mu_and_sigma[0], category_averg_mu_and_sigma[1]
    if not single_data:
        return category_mu, category_sigm

    mu, sigm = norm.fit(single_data)
    if not sigm:
        sigm = category_averg_mu_and_sigma[1]

    mu = mu_param_lambda * mu + (1 - mu_param_lambda) * category_mu
    sigm = sigm_param_lambda * sigm + (1 - sigm_param_lambda) * category_sigm

    return mu, sigm

def average_sigm_from_more_than_one(data):
    mu_sum = 0.0
    sigma_sum = 0.0
    mu_num = 0
    sigma_num = 0
    result = 0.0
    for ix in xrange(len(data)):
        value = data[ix]
        if len(set(value)) > 0:
            mu, sigm = norm.fit(value)
            mu_sum += mu
            mu_num += 1
            if len(set(value)) > 1:
                sigma_sum += sigm
                sigma_num += 1
    result = (mu_sum / mu_num, sigma_sum / sigma_num)
    return result


def gaussian_curve_fit(data):
    row_num = len(data)
    result_mu = np.array([0 for _ in range(row_num)])
    result_sigm = np.array([0 for _ in range(row_num)])

    averg_mu_and_sigm = average_sigm_from_more_than_one(data)

    for ix in range(row_num):
        result_mu[ix], result_sigm[ix] = _gaussian_curve_fit(data[ix], averg_mu_and_sigm)

    return result_mu, result_sigm

def get_prob_from_gaussian(price, mu, sigm):
    prob = np.exp(-(price-mu)**2/(2*sigm**2)) / (sigm * np.sqrt(2 * np.pi))
    return prob
