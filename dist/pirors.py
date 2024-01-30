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
        # NOT including the identity matrix, but with intercept
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
        self.data = self.likelihood['Y']

    def _parse_params(self):
        self.lag = self.params['lag']
        self.lambda1 = self.params['lambda1']
        self.lambda2 = self.params['lambda2']
        self.lambda3 = self.params['lambda3']
        self.lambda4 = self.params['lambda4']

    def calc_comp_param(self, sigma):
        pass

    def calc_cov_param(self):
        pass


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
class_factory = {}


def register_class(cls):
    class_factory[cls.__name__] = cls


register_class(DiffusePrior)
register_class(MinnesotaPrior)
register_class(SteadyStatePrior)
