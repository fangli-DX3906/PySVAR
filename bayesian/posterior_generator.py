import numpy as np

from dist.distributions import *
from dist.priors import *


class PosteriorGenerator:
    def __init__(self,
                 prior_name: str,
                 likelihood_info: dict,
                 prior_params: dict):

        self.likelihood_info = likelihood_info
        self.prior = prior_dist_registry[prior_name](likelihood_info=self.likelihood_info, params=prior_params)
        self._parse_likelihood_info()

    def _parse_likelihood_info(self):
        self.n = self.likelihood_info['n']
        self.p = self.likelihood_info['p']
        self.constant = self.likelihood_info['constant']

    def _recover_comp_mat(self, comp: np.ndarray):
        id = np.eye(self.n * self.p)
        comp = comp.reshape((-1, self.n), order='F')
        if self.constant:
            comp = np.concatenate((comp[1:, :].T, id[:-self.n, :]), axis=0)
        else:
            comp = np.concatenate((comp.T, id[:-self.n, :]), axis=0)
        return comp

    def draw_from_posterior(self, **kwargs):
        comp_param = self.prior.calc_posterior_comp_param(**kwargs)

        if len(comp_param) == 4:
            B_tilde = comp_param[2]
            omega_tilde = comp_param[3]
            comp_param = comp_param[:2]
        else:
            B_tilde = None
            omega_tilde = None

        B = MultiNormalDist(*comp_param)()
        B = self._recover_comp_mat(B)

        cov_param = self.prior.calc_posterior_cov_param(B=B, omega_tilde=omega_tilde, B_tilde=B_tilde)
        sigma = InverseWhishartDist(*cov_param)()

        return B, sigma
