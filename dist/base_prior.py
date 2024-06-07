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
        self.T = self.likelihood_info['T']
        self.n = self.likelihood_info['n']
        self.p = self.likelihood_info['p']
        self.Y = self.likelihood_info['Y']
        self.y = self.likelihood_info['y']
        self.X = self.likelihood_info['X']
        self.Bhat = self.likelihood_info['Bhat']
        self.sigma = self.likelihood_info['sigma']
        self.constant = self.likelihood_info['constant']
        self.resids = self.likelihood_info['resids']
        
        self.b = 1 if self.constant else 0
        self.np = self.n * self.p + self.b

    def __repr__(self):
        return str(self.__class__).split('.')[-1][:-2]
