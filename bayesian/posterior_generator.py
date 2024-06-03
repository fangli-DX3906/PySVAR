import numpy as np

from dist.distributions import *
from dist.pirors import *


class PosteriorGenerator:
    def __init__(self,
                 likelihood_info: dict,
                 prior_name: str,
                 prior_params: dict):
        self.likelihood = likelihood_info
        self._parse_likelihood_info()
        self.prior = prior_dist_registry[prior_name](likelihood=self.likelihood, params=prior_params)

    def _calc_posterior_dist_param(self, *args, **kwargs):
        comp_param = self.prior.calc_comp_param(kwargs['sigma'])
        cov_param = self.prior.calc_cov_param()
        return comp_param, cov_param

    def _parse_likelihood_info(self):
        self.n = self.likelihood['n']
        self.p = self.likelihood['p']
        self.constant = self.likelihood['const']

    def _recover_comp_mat(self,
                          comp: np.ndarray):
        id = np.eye(self.n * self.p)
        comp = comp.reshape((-1, self.n), order='F')
        if self.constant:
            comp = np.concatenate((comp[1:, :].T, id[:-self.n, :]), axis=0)
        else:
            comp = np.concatenate((comp.T, id[:-self.n, :]), axis=0)
        return comp

    def draw_from_posterior(self, *args, **kwargs):
        comp_param, cov_param = self._calc_posterior_dist_param(*args, **kwargs)
        B = MultiNormalDist(*comp_param)()
        B = self._recover_comp_mat(B)
        sigma = InverseWhishartDist(*cov_param)()
        return B, sigma
