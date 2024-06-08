import numpy as np
from base_prior import PriorDist
from typing import Optional

from utils.funcs import calc_ar1_coeff

# which to import
__all__ = ['DiffusePrior', 'NormalDiffusePrior',
           'MinnesotaPrior', 'NaturalConjugatePrior',
           'prior_dist_registry']

# register
prior_dist_registry = {}


def register_class(cls):
    prior_dist_registry[cls.__name__] = cls


########################################################################################################################
class DiffusePrior(PriorDist):
    def __init__(self,
                 likelihood_info: dict,
                 params: None):
        self.likelihood_info = likelihood_info
        self._parse_likelihood_info()

    def calc_posterior_comp_param(self, **kwargs):
        sigma = kwargs['sigma']
        p_mn_1 = self.Bhat.reshape((-1, 1), order='F')
        p_mn_2 = np.kron(sigma, np.linalg.inv(np.dot(self.X.T, self.X)))
        return p_mn_1, p_mn_2

    def calc_posterior_cov_param(self, **kwargs):
        p_iw_1 = self.sigma * self.T
        p_iw_2 = self.T - self.n
        return p_iw_1, p_iw_2


class MinnesotaPrior(PriorDist):
    """

        Minnesota prior should be passed a parameter dictionary including the hyperparameters:
            \lambda_1 controls the standard deviation of the prior on own lags
            \lambda_2 controls the standard deviation of the prior on lags of other variables
            \lambda_3 controls the speed of higher lags shrinking to zero
            \lambda_4 controls the prior variance on the constant

            comp_mode controls how the coefficient prior mean is set: 'RandomWalk' or 'AR1'

            if cov_mode is set to 'Strict', then scalar for IW distribution should be passed thought cov_scalar

    """

    def __init__(self,
                 likelihood_info: dict,
                 params: dict):

        self.likelihood_info = likelihood_info
        self._parse_likelihood_info()

        self.ar1_coeff, self.ar1_sigma = calc_ar1_coeff(self.Y)

        self.params = params
        self.calc_prior_param()

    def calc_prior_param(self):
        self.comp_mode = self.params.get('comp_mode', 'RandomWalk')
        if self.comp_mode == 'RandomWalk':
            self.ar1_coeff = [1] * self.n

        self.iw_1 = self.params.get('cov_scalar', np.eye(self.n))
        self.iw_2 = self.n + 1

        self.lam1 = self.params.get('lambda1', 0.2)
        self.lam2 = self.params.get('lambda2', 0.5)
        self.lam3 = self.params.get('lambda3', 1)
        self.lam4 = self.params.get('lambda4', 10 ** 5)

        self.mn_1, self.mn_2 = self.calc_prior_comp_param()

    def calc_prior_comp_param(self):
        B0 = np.zeros_like(self.Bhat)

        for i in range(self.n):
            B0[i + self.b, i] = self.ar1_coeff[i]

        mn_2 = np.zeros((self.n * self.np, self.n * self.np))
        for i in range(self.n):
            scalar = self.ar1_sigma[i] ** 2
            if self.constant:
                li = [self.lam4 ** 2]
            else:
                li = []
            for p in range(self.p):
                for j in range(self.n):
                    item = self.lam1 ** 2 / ((p + 1) ** self.lam3 * self.ar1_sigma[j]) ** 2
                    if i != j:
                        item *= self.lam2 ** 2
                    li.append(item)
            mn_2[i * self.np:(i + 1) * self.np, i * self.np:(i + 1) * self.np] = scalar * np.diag(li)

        mn_1 = B0.reshape((-1, 1), order='F')

        return mn_1, mn_2

    def calc_posterior_comp_param(self, **kwargs):
        sigma = kwargs['sigma']
        bhat = self.Bhat.reshape((-1, 1), order='F')

        mat1 = np.linalg.inv(self.mn_2) + np.kron(np.linalg.inv(sigma), np.dot(self.X.T, self.X))
        mat21 = np.dot(np.linalg.inv(self.mn_2), self.mn_1)
        mat22 = np.dot(np.kron(np.linalg.inv(sigma), np.dot(self.X.T, self.X)), bhat)
        mat2 = mat21 + mat22
        p_mn_1 = np.dot(np.linalg.inv(mat1), mat2)
        p_mn_2 = np.linalg.inv(mat1)

        return p_mn_1, p_mn_2

    def calc_posterior_cov_param(self, **kwargs):
        B = kwargs['B']
        B = B.reshape((-1, self.n), order='F')
        resids = self.y - np.dot(self.X, B)
        p_iw_1 = self.iw_1 + np.dot(resids.T, resids)
        p_iw_2 = self.iw_2 + self.T
        return p_iw_1, p_iw_2


class NaturalConjugatePrior(PriorDist):
    """

            Natural conjugate prior should be passed a parameter dictionary including the hyperparameters:
                if comp_mode is set to 'RandomWalk' or 'AR1', then need to pass the following params
                    \lambda_1 controls the standard deviation of the prior on own lags
                    \lambda_2 is set to 1
                    \lambda_3 controls the speed of higher lags shrinking to zero
                    \lambda_4 controls the prior variance on the constant

                else comp_mode is set to 'Set', then need to provide the prior parameters for MN and IW

        """

    def __init__(self,
                 likelihood_info: dict,
                 params: Optional[dict]):

        self.likelihood_info = likelihood_info
        self._parse_likelihood_info()

        self.ar1_coeff, self.ar1_sigma = calc_ar1_coeff(self.Y)

        self.params = params
        self.calc_prior_param()

    def calc_prior_param(self):
        self.comp_mode = self.params.get('comp_mode', 'RandomWalk')

        if self.comp_mode != 'Set':
            assert self.comp_mode in ['RandomWalk', 'AR1']
            self.lam1 = self.params.get('lambda1', 0.2)
            self.lam3 = self.params.get('lambda3', 1)
            self.lam4 = self.params.get('lambda4', 10 ** 5)
            self.iw_2 = self.n + 1
            self.iw_1 = np.diag([i ** 2 for i in self.ar1_sigma])
            if self.comp_mode == 'RandomWalk':
                self.ar1_coeff = [1] * self.n
            self.mn_1, self.mn_2 = self.calc_prior_comp_param()
        else:
            self.mn_1 = self.params['mn_mean']
            self.mn_1.reshape((-1, 1), order='F')
            self.mn_2 = self.params['mn_cov']
            self.iw_1 = self.params['iw_scalar']
            self.iw_2 = self.params['iw_df']
            if self.iw_2 < self.n:
                raise ValueError('Degree of freedom of IW dist is too small.')

    def calc_prior_comp_param(self):
        B0 = np.zeros_like(self.Bhat)

        for i in range(self.n):
            B0[i + self.b, i] = self.ar1_coeff[i]

        if self.constant:
            li = [self.lam4 ** 2]
        else:
            li = []
        for p in range(self.p):
            for j in range(self.n):
                item = self.lam1 ** 2 / ((p + 1) ** self.lam3 * self.ar1_sigma[j]) ** 2
                li.append(item)
        mn_2 = np.diag(li)
        mn_1 = B0.reshape((-1, 1), order='F')

        return mn_1, mn_2

    def calc_posterior_comp_param(self, **kwargs):
        sigma = kwargs['sigma']
        omega_tilde = np.linalg.inv(np.linalg.inv(self.mn_2) + np.dot(self.X.T, self.X))
        mat1 = np.dot(np.linalg.inv(self.mn_2), self.mn_1.reshape((-1, self.n), order='F'))
        mat2 = np.dot(np.dot(self.X.T, self.X), self.Bhat)
        B_tilde = np.dot(omega_tilde, mat1 + mat2)
        p_mn_1 = B_tilde.reshape((-1, 1), order='F')
        p_mn_2 = np.kron(sigma, omega_tilde)

        return p_mn_1, p_mn_2, B_tilde, omega_tilde

    def calc_posterior_cov_param(self, **kwargs):
        B_tilde = kwargs['B_tilde']
        omega_tilde = kwargs['omega_tilde']

        mat1 = np.dot(np.dot(self.Bhat.T, np.dot(self.X.T, self.X)), self.Bhat)
        B_bar = self.mn_1.reshape((-1, self.n), order='F')
        mat2 = np.dot(np.dot(B_bar.T, np.linalg.inv(self.mn_2)), B_bar)
        mat3 = np.dot(self.resids, self.resids.T)
        mat4 = np.dot(np.dot(B_tilde.T, np.linalg.inv(omega_tilde)), B_tilde)
        p_iw_1 = mat1 + mat2 + self.iw_1 + mat3 - mat4
        p_iw_2 = self.T + self.iw_2

        return p_iw_1, p_iw_2


class NormalDiffusePrior(PriorDist):
    """

        Normal diffuse prior should be passed a parameter dictionary including the hyperparameters:
            if comp_mode is set to 'RandomWalk' or 'AR1', then need to pass the following params
                \lambda_1 controls the standard deviation of the prior on own lags
                \lambda_2 is set to 1
                \lambda_3 controls the speed of higher lags shrinking to zero
                \lambda_4 controls the prior variance on the constant

            else comp_mode is set to 'Set', then need to provide the prior parameters for MN

    """

    def __init__(self,
                 likelihood_info: dict,
                 params: Optional[dict]):

        self.likelihood_info = likelihood_info
        self._parse_likelihood_info()

        self.ar1_coeff, self.ar1_sigma = calc_ar1_coeff(self.Y)

        self.params = params
        self.calc_prior_param()

    def calc_prior_param(self):
        self.comp_mode = self.params.get('comp_mode', 'RandomWalk')

        if self.comp_mode != 'Set':
            assert self.comp_mode in ['RandomWalk', 'AR1']
            self.lam1 = self.params.get('lambda1', 0.2)
            self.lam3 = self.params.get('lambda3', 1)
            self.lam4 = self.params.get('lambda4', 10 ** 5)
            if self.comp_mode == 'RandomWalk':
                self.ar1_coeff = [1] * self.n
            self.mn_1, self.mn_2 = self.calc_prior_comp_param()
        else:
            self.mn_1 = self.params['mn_mean']
            self.mn_1.reshape((-1, 1), order='F')
            self.mn_2 = self.params['mn_cov']

    def calc_prior_comp_param(self):
        B0 = np.zeros_like(self.Bhat)

        for i in range(self.n):
            B0[i + self.b, i] = self.ar1_coeff[i]

        mn_2 = np.zeros((self.n * self.np, self.n * self.np))
        for i in range(self.n):
            scalar = self.ar1_sigma[i] ** 2
            if self.constant:
                li = [self.lam4 ** 2]
            else:
                li = []
            for p in range(self.p):
                for j in range(self.n):
                    item = self.lam1 ** 2 / ((p + 1) ** self.lam3 * self.ar1_sigma[j]) ** 2
                    li.append(item)
            mn_2[i * self.np:(i + 1) * self.np, i * self.np:(i + 1) * self.np] = scalar * np.diag(li)
        mn_1 = B0.reshape((-1, 1), order='F')

        return mn_1, mn_2

    def calc_posterior_comp_param(self, **kwargs):
        sigma = kwargs['sigma']
        mat1 = np.kron(np.linalg.inv(sigma), np.dot(self.X.T, self.X))
        p_mn_2 = np.linalg.inv(np.linalg.inv(self.mn_2) + mat1)
        mat21 = np.dot(np.linalg.inv(self.mn_2), self.mn_1)
        mat22 = np.dot(mat1, self.Bhat.reshape((-1, 1), order='F'))
        B_tilde = np.dot(p_mn_2, mat21 + mat22)
        p_mn_1 = B_tilde.reshape((-1, 1), order='F')

        return p_mn_1, p_mn_2

    def calc_posterior_cov_param(self, **kwargs):
        B = kwargs['B']
        B = B.reshape((-1, self.n), order='F')
        mat1 = np.dot(self.resids, self.resids.T)
        mat2 = np.dot(np.dot((B - self.Bhat).T, np.dot(self.X.T, self.X)), (B - self.Bhat))
        p_iw_1 = mat1 + mat2
        p_iw_2 = self.T

        return p_iw_1, p_iw_2


########################################################################################################################
# register all prior classes
register_class(DiffusePrior)
register_class(MinnesotaPrior)
register_class(NaturalConjugatePrior)
register_class(NormalDiffusePrior)
