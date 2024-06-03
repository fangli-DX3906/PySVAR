from abc import ABCMeta, abstractmethod
import numpy as np


class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self.__class__).split('.')[-1][:-2]


class InverseWhishartDist(Distribution):
    def __init__(self,
                 s: np.ndarray,
                 nu: np.ndarray,
                 check: bool = False):
        self.s = s
        self.nu = nu
        if check:
            if np.linalg.det(self.s) < 0:
                raise ValueError(f's is not positive definite')

    def __call__(self):
        S = np.linalg.inv(self.s)
        A = np.dot(np.linalg.cholesky(S), np.random.randn(S.shape[0], self.nu))
        return np.linalg.inv(np.dot(A, A.T))


class MultiNormalDist(Distribution):
    def __init__(self,
                 psi: np.ndarray,
                 omega: np.ndarray):
        self.psi = psi
        self.omega = omega

    def __call__(self):
        n_params = self.psi.shape[0]
        A = self.psi + np.dot(np.linalg.cholesky(self.omega), np.random.randn(n_params, 1))
        return A


# which to import
__all__ = ['InverseWhishartDist', 'MultiNormalDist']
