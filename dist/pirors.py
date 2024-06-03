from abc import ABCMeta, abstractmethod
import numpy as np


class PriorDist(metaclass=ABCMeta):
    @abstractmethod
    def calc_comp_param(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def calc_cov_param(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _parse_likelihood_info(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self.__class__).split('.')[-1][:-2]


class DiffusePrior(PriorDist):
    def __init__(self,
                 likelihood: dict,
                 params: None):
        self.likelihood = likelihood
        self._parse_likelihood_info()

    def _parse_likelihood_info(self):
        self.n = self.likelihood['n']
        self.p = self.likelihood['p']
        self.X = self.likelihood['X']
        self.Bhat = self.likelihood['Bhat']
        self.t = self.likelihood['t']
        self.n = self.likelihood['n']
        self.sigma = self.likelihood['sigma']

    def calc_comp_param(self, sigma):
        mn_1 = self.Bhat.reshape((-1, 1), order='F')
        mn_2 = np.kron(sigma, np.linalg.inv(np.dot(self.X.T, self.X)))
        return mn_1, mn_2

    def calc_cov_param(self):
        iw_1 = self.sigma * self.t
        iw_2 = self.t - self.n
        return iw_1, iw_2


class MinnesotaPrior(PriorDist):
    def __init__(self,
                 likelihood: dict,
                 params: dict):
        self.likelihood = likelihood
        self.params = params
        self._parse_likelihood_info()
        self._parse_params()

    def _parse_likelihood_info(self):
        self.n = self.likelihood['n']
        self.p = self.likelihood['p']
        self.Y = self.likelihood['Y']
        self.X = self.likelihood['X']
        self.sigma = self.likelihood['sigma']
        self.Bhat = self.likelihood['Bhat']

    def _parse_params(self):
        self.lag = self.params['lag']
        self.lambda1 = self.params['lambda1']
        self.lambda2 = self.params['lambda2']
        self.lambda3 = self.params['lambda3']
        self.lambda4 = self.params['lambda4']

    def calc_comp_param(self, sigma):
        if self.p != self.Y.shape[1]:
            raise ValueError
        list_temp = []
        for i in range(self.n):
            for j in range(self.lag * self.n + 1):
                if j == i + 1:
                    temp = self.Bhat[j - 1]
                else:
                    temp = 0
                list_temp.append(temp)
        self.psi = list_temp

        # transform the vector into a matrix
        self.psi_bar = np.reshape(self.psi, (-1, self.n * self.lag + 1)).T

        # construct the H matrix
        list2 = [self.lambda4 ** 2]
        for i in range(1, self.lag + 1):
            for j in range(0, self.n):
                temp = (self.lambda1 / np.sqrt(self.sigma[j, j]) / i ** self.lambda3) ** 2
                list2.append(temp)

        self.omega_bar = np.diag(list2)
        self.omega_tilde = np.linalg.inv(np.linalg.inv(self.omega_bar) + np.dot(self.X.T, self.X))
        self.psi_tilde = np.dot(self.omega_tilde,
                                np.dot(self.X.T, self.Y) + np.dot(np.linalg.inv(self.omega_bar), self.psi_bar))
        self.mn_1 = np.array(np.reshape(self.psi_tilde.T, (-1, 1)))
        self.mn_2 = np.kron(np.diag(np.diag(self.sigma)), self.omega_tilde)
        return self.mn_1, self.mn_2

    def calc_cov_param(self):
        self.nu_bar = self.n + 1
        self.nu_tilde = self.nu_bar + self.n
        self.phi_bar = np.eye(self.n)
        self.phi_tilde = np.dot(self.Y.T, self.Y) + self.phi_bar + \
                         np.dot(np.dot(self.psi_bar.T, np.linalg.inv(self.omega_bar)), self.psi_bar) - \
                         np.dot(np.dot(self.psi_tilde.T, np.linalg.inv(self.omega_tilde)), self.psi_tilde)
        self.iw_1 = self.nu_tilde
        self.iw_2 = self.phi_tilde
        return self.iw_1, self.iw_2


class SteadyStatePrior(PriorDist):
    def __init__(self,
                 likelihood: dict,
                 params: dict):
        self.likelihood = likelihood
        self.params = params
        self._parse_likelihood_info()
        self._parse_params()

    def _parse_likelihood_info(self):
        pass

    def _parse_params(self):
        pass

    def calc_comp_param(self, sigma):
        pass

    def calc_cov_param(self):
        pass


# register
prior_dist_registry = {}


def register_class(cls):
    prior_dist_registry[cls.__name__] = cls


register_class(DiffusePrior)
register_class(MinnesotaPrior)
register_class(SteadyStatePrior)

# which to import
__all__ = ['DiffusePrior', 'MinnesotaPrior', 'SteadyStatePrior', 'prior_dist_registry']
