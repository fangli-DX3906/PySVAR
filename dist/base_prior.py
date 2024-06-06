from abc import ABCMeta, abstractmethod


# base prior class
class PriorDist(metaclass=ABCMeta):
    @abstractmethod
    def calc_posterior_comp_param(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def calc_posterior_cov_param(self, *args, **kwargs):
        raise NotImplementedError

    def _parse_likelihood_info(self):
        self.t = self.likelihood_info['t']
        self.n = self.likelihood_info['n']
        self.lag = self.likelihood_info['p']
        self.Y = self.likelihood_info['Y']
        self.X = self.likelihood_info['X']
        self.Bhat = self.likelihood_info['Bhat']
        self.sigma = self.likelihood_info['sigma']
        self.constant = self.likelihood_info['const']
        self.resids = self.likelihood_info['resids']

    def __repr__(self):
        return str(self.__class__).split('.')[-1][:-2]
