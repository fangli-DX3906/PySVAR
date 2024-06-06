import numpy as np
from base_prior import PriorDist
from typing import Optional
from statsmodels.api import OLS, add_constant

# which to import
__all__ = ['DiffusePrior', 'NormalDiffusePrior',
           'MinnesotaPrior', 'NaturalConjugatePrior',
           'prior_dist_registry']

# register
prior_dist_registry = {}


def register_class(cls):
    prior_dist_registry[cls.__name__] = cls


def calc_ar1_coeff(data: np.ndarray):
    ar1_coeff = []
    ar1_sigma = []
    for i in range(data.shape[1]):
        reg = OLS(data[1:, i], add_constant(data[:-1, i])).fit()
        ar1_coeff.append(reg.params[1])
        ar1_sigma.append(reg.mse_resid ** 0.5)

    return ar1_coeff, ar1_sigma


###########################################
class DiffusePrior(PriorDist):
    def __init__(self,
                 likelihood_info: dict,
                 params: None):
        self.likelihood_info = likelihood_info
        self._parse_likelihood_info()

    def calc_posterior_comp_param(self, sigma):
        p_mn_1 = self.Bhat.reshape((-1, 1), order='F')
        p_mn_2 = np.kron(sigma, np.linalg.inv(np.dot(self.X.T, self.X)))
        return p_mn_1, p_mn_2

    def calc_posterior_cov_param(self):
        p_iw_1 = self.sigma * self.t
        p_iw_2 = self.t - self.n
        return p_iw_1, p_iw_2


class MinnesotaPrior(PriorDist):
    """

        Minnesota prior should be passed a parameter dictionary including the hyperparameters:
            \lambda_1 controls the standard deviation of the prior on own lags
            \lambda_2 controls the standard deviation of the prior on lags of other variables
            \lambda_3 controls the speed of higher lags shrinking to zero
            \lambda_4 controls the prior variance on the constant

            comp_mode controls how the coefficient prior mean is set: 'RandomWalk' or 'AR1'
            cov_mode controls if the covariance matrix is set known: 'Theil' or 'Strict

    """

    def __init__(self,
                 likelihood_info: dict,
                 params: dict):

        self.likelihood_info = likelihood_info
        self._parse_likelihood_info()

        self.params = params
        self._parse_dist_param()

        self.ar1_coeff, self.ar1_sigma = calc_ar1_coeff(self.Y)
        self.b = 1 if self.constant else 0
        self.np = self.n * self.lag + self.b

    def _parse_dist_param(self):
        self.mode = self.params['mode']  # taking either 'AR1' or 'RandomWalk'
        self.lam1 = self.params.get('lambda1', 0.2)
        self.lam2 = self.params.get('lambda2', 0.5)
        self.lam3 = self.params.get('lambda3', 1)
        self.lam4 = self.params.get('lambda4', 10 ** 5)

    def calc_prior_comp_param(self):
        B0 = np.zeros_like(self.Bhat)

        if self.mode == 'AR1':
            for i in range(self.n):
                B0[i + self.b, i] = self.ar1_coeff[i]
        else:
            for i in range(self.n):
                B0[i + self.b, i] = 1

        mn_2 = np.zeros(self.n * self.np)
        for i in range(self.n):
            scalar = self.ar1_sigma[i] ** 2
            if self.constant:
                li = [self.lam4 ** 2]
            else:
                li = []
            for p in range(self.lag):
                for j in range(self.n):
                    item = self.lam1 ** 2 / ((p + 1) ** self.lam3 * self.ar1_sigma[j]) ** 2
                    if i != j:
                        item *= self.lam2 ** 2
                    li.append(item)
            mn_2[i * self.np:(i + 1) * self.np, i * self.np:(i + 1) * self.np] = scalar * np.diag(li)

        mn_1 = B0.reshape((-1, 1), order='F')

        return mn_1, mn_2

    def calc_prior_cov_param(self):
        iw_1 = np.eye(self.n)
        iw_2 = self.n + 1

        return iw_1, iw_2

    def calc_posterior_comp_param(self):
        mn_1, mn_2 = self.calc_prior_comp_param()
        sigma = np.eye(self.n)
        bhat = self.Bhat.reshape((-1, 1), order='F')
        mat1 = np.linalg.inv(mn_2) + np.kron(np.linalg.inv(sigma), np.dot(self.X.T, self.X))
        mat21 = np.dot(np.linalg.inv(mn_2), mn_1)
        mat22 = np.kron(np.linalg.inv(sigma), np.dot(np.dot(self.X.T, self.X), bhat))
        mat2 = mat21 + mat22
        p_mn_1 = np.dot(np.linalg.inv(mat1), mat2)
        p_mn_2 = np.linalg.inv(mat1)

        return p_mn_1, p_mn_2

    def calc_posterior_cov_param(self):
        iw_1, iw_2 = self.calc_prior_cov_param()
        p_iw_1 = iw_1 + np.dot(self.resids, self.resids.T)
        p_iw_2 = iw_2 + self.t

        return p_iw_1, p_iw_2


class NaturalConjugatePrior(PriorDist):
    def __init__(self,
                 likelihood_info: dict,
                 params: Optional[dict]):

        self.likelihood_info = likelihood_info
        self.params = params
        self._parse_likelihood_info()
        self._parse_dist_param()
        self.ar1_coeff, self.ar1_sigma = calc_ar1_coeff(self.Y)
        self.b = 1 if self.constant else 0
        self.np = self.n * self.lag + self.b

    def _parse_dist_param(self):
        self.mode = self.params['mode']  # taking either 'ar1' or 'rw' or 'set'
        if self.mode == 'ar1' or self.mode == 'rw':
            self.lam1 = self.params.get('lambda1', 0.2)
            self.lam3 = self.params.get('lambda3', 1)
            self.lam4 = self.params.get('lambda4', 10 ** 5)
            self.iw_2 = self.params.get('iw_df', self.n + 1)
        else:
            assert self.mode == 'set'
            self.mn_1 = self.params['mn_mean']
            self.mn_2 = self.params['mn_cov']
            self.iw_1 = self.params['iw_scalar']
            self.iw_2 = self.params['iw_df']

    def calc_prior_comp_param(self):
        mn_1 = np.zeros_like(self.Bhat)

        if self.mode != 'set':
            if self.mode == 'ar1':
                for i in range(self.n):
                    mn_1[i, i + self.b] = self.ar1_coeff[i]
            elif self.mode == 'rw':
                for i in range(self.n):
                    mn_1[i, i + self.b] = 1
            mn_2 = np.zeros(self.n * self.np)
            for i in range(self.n):
                if self.constant:
                    li = [self.lam4 ** 2]
                else:
                    li = []
                for p in range(self.lag):
                    for j in range(self.n):
                        item = self.lam1 ** 2 / ((p + 1) ** self.lam3 * self.ar1_sigma[j]) ** 2
                        li.append(item)
                mn_2[i * self.np:(i + 1) * self.np, i * self.np:(i + 1) * self.np] = np.diag(li)
        else:
            mn_1 = self.mn_1
            mn_2 = self.mn_2

        return mn_1, mn_2

    def calc_prior_cov_param(self):
        iw_1 = np.eye(self.n)
        for i in range(self.n):
            iw_1[i, i] = self.ar1_sigma[i] ** 2

        return iw_1, self.iw_2

    def calc_posterior_comp_param(self):
        mn_1, mn_2 = self.calc_prior_comp_param()
        omega_tilde = np.linalg.inv(np.linalg.inv(mn_2) + np.dot(self.X.T, self.X))
        B_tilde = np.dot(mn_2, np.dot(np.linalg.inv(mn_2), self.mn_1.T) + np.dot(np.dot(self.X.T, self.X), self.Bhat.T))
        p_mn_1 = B_tilde.T.reshape((-1, 1), order='F')
        p_mn_2 = np.kron(self.sigma, omega_tilde)

        return p_mn_1, p_mn_2

    def calc_posterior_cov_param(self):
        mn_1, mn_2 = self.calc_prior_comp_param()
        iw_1, p_iw_2 = self.calc_prior_cov_param()
        B_tilde = np.dot(mn_2, np.dot(np.linalg.inv(mn_2), self.mn_1.T) + np.dot(np.dot(self.X.T, self.X), self.Bhat.T))
        omega_tilde = np.linalg.inv(mn_2) + np.dot(self.X.T, self.X)
        mat1 = np.dot(np.dot(self.Bhat, np.dot(self.X.T, self.X)), self.Bhat.T)
        mat2 = np.dot(np.dot(mn_1, np.linalg.inv(mn_2)), mn_1.T)
        mat3 = np.dot(self.resids, self.resids.T)
        mat4 = np.dot(np.dot(B_tilde.T, omega_tilde), B_tilde)
        p_iw_1 = mat1 + mat2 + iw_1 + mat3 - mat4

        return p_iw_1, p_iw_2


class NormalDiffusePrior(PriorDist):
    pass


# register all prior classes
register_class(DiffusePrior)
register_class(MinnesotaPrior)
register_class(NaturalConjugatePrior)
register_class(NormalDiffusePrior)
