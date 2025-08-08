import numpy as np
import scipy.optimize as spo
from typing import Literal, Optional, Set, overload

from numpy.core import overrides

from core.svar import PointIdentifiedSVAR


class ExclusionRestriction(PointIdentifiedSVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 exclusion: Set[tuple],
                 constant: bool = True,
                 lag_order: Optional[int] = None,
                 max_lag_order: Optional[int] = 8,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 date_frequency: Literal['M', 'Q', 'A'] = 'Q',
                 date_start: Optional[str] = None):

        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         constant=constant,
                         lag_order=lag_order,
                         max_lag_order=max_lag_order,
                         info_criterion=info_criterion,
                         date_frequency=date_frequency,
                         date_start=date_start)

        self.identification = 'exclusion identification'
        self.exclusions = exclusion
        self.rotation = None
        self.n_restrictions = len(exclusion)
        if self.n_restrictions != self.n_vars * (self.n_vars - 1) / 2:
            raise ValueError('The model is not exactly identified!')

        self.all_list = {(i, j) for i in range(self.n_vars) for j in range(self.n_vars)}

    def target_function(self, A: np.ndarray, cov_mat: np.ndarray):
        A = A.reshape((self.n_vars, -1))
        func = np.dot(A, A.T) - cov_mat
        func_sum = 0
        for idx in self.exclusions:
            func_sum += A[idx[0], idx[1]] ** 2
        func = np.sum(func ** 2) + func_sum
        return func

    def solve(self,
              comp_mat: Optional[np.ndarray] = None,
              cov_mat: Optional[np.ndarray] = None) -> np.ndarray:
        if cov_mat is None:
            cov_mat = self.cov_mat
        target_func = lambda A: self.target_function(A, cov_mat=cov_mat)
        sol = spo.minimize(fun=target_func, x0=np.eye(self.n_vars))
        rotation = sol.x.reshape((self.n_vars, -1))
        return rotation
