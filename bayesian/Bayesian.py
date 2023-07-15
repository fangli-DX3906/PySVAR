from utils.distributions import *

import numpy as np


class Bayesian:
    def __init__(self, likelihood_info: dict):
        self.likelihood_info = likelihood_info
        self.n = self.likelihood_info['n']
        self.p = self.likelihood_info['p']

    def draw_comp_from_posterior(self, mn_1, mn_2):
        B = draw_multi_normal_distribution(mn_1, mn_2)
        # remove the intercept but add back the identity matrix
        B = self.__recover_comp(B)
        return B

    def draw_cov_from_posterior(self, iw_1, iw_2):
        return draw_inverse_whishart_distribution(iw_1, iw_2)

    def __posterior_comp_dist_params_ss_piror(self):
        pass

    def __posterior_cov_dist_params_ss_piror(self):
        pass

    def __recover_comp(self, comp):
        id = np.eye(self.n * self.p)
        comp = comp.reshape((-1, self.n), order='F')
        comp = np.concatenate((comp[1:, :].T, id[:-self.n, :]), axis=0)
        return comp


class DiffusePrior(Bayesian):
    def __init__(self, likelihood_info: dict):
        super().__init__(likelihood_info)

    def get_posterior_comp_dist_params(self, sigma):
        return self.__posterior_comp_dist_params_diffuse_prior(sigma)

    def get_posterior_cov_dist_params(self):
        return self.__posterior_cov_dist_params_diffuse_prior()

    def __posterior_comp_dist_params_diffuse_prior(self, sigma):
        X = self.likelihood_info['X']
        # NOT including the identity matrix, but with intercept
        mn_1 = self.likelihood_info['Bhat'].reshape((-1, 1), order='F')
        mn_2 = np.kron(sigma, np.linalg.inv(np.dot(X.T, X)))
        return mn_1, mn_2

    def __posterior_cov_dist_params_diffuse_prior(self):
        t = self.likelihood_info['t']
        n = self.likelihood_info['n']
        iw_1 = self.likelihood_info['sigma'] * t
        iw_2 = t - n
        return iw_1, iw_2


class MinnesotaPrior(Bayesian):
    def __init__(self,
                 data: np.ndarray,
                 likelihood_info: dict,
                 minnesota_params: dict):
        super().__init__(likelihood_info)
        self.lag = minnesota_params['lag']
        self.lambda1 = minnesota_params['lambda1']
        self.lambda2 = minnesota_params['lambda2']
        self.lambda3 = minnesota_params['lambda3']
        self.lambda4 = minnesota_params['lambda4']
        self.data = data

    # TODO:
    def get_posterior_comp_dist_params(self, sigma):
        pass

    # TODO:
    def get_posterior_cov_dist_params(self):
        pass

    def __posterior_comp_dist_params_diffuse_prior(self, sigma):
        pass

    def __posterior_cov_dist_params_diffuse_prior(self):
        pass
