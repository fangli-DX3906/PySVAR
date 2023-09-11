import numpy as np
import scipy.optimize as spo
from typing import Literal, Optional, Set
import datetime

from estimation.SVAR import PointIdentifiedSVAR


class ExclusionRestriction(PointIdentifiedSVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 exclusion: Set[tuple],
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 date_start: datetime.datetime,
                 date_end: datetime.datetime,
                 lag_order: Optional[int] = None,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         date_frequency=date_frequency,
                         date_start=date_start,
                         date_end=date_end,
                         lag_order=lag_order,
                         constant=constant,
                         info_criterion=info_criterion)
        self.exclusions = exclusion
        self.rotation = None
        self.n_restrictions = len(exclusion)
        if self.n_restrictions != self.n_vars * (self.n_vars - 1) / 2:
            raise ValueError('The model is not exactly identified!')
        all_list = {(i, j) for i in range(self.n_vars) for j in range(self.n_vars)}
        self.unrestricted = all_list.difference(self.exclusions)

    def assign_non_zero_elements(self, unknown: np.ndarray) -> np.ndarray:
        target_rotation = np.zeros((self.n_vars, self.n_vars))
        for idx, item in enumerate(self.unrestricted):
            target_rotation[item] = unknown[idx]
        return target_rotation

    def make_target_function(self,
                             x: np.ndarray,
                             cov_mat: np.ndarray,
                             normalization: bool = True):
        B_mat = self.assign_non_zero_elements(x)
        rotation = np.linalg.inv(B_mat)
        if normalization:
            func = np.dot(rotation, rotation.T) - cov_mat
        else:
            func = 0
        return np.sum(func ** 2)

    def solve(self,
              comp_mat: Optional[np.ndarray] = None,
              cov_mat: Optional[np.ndarray] = None):
        if cov_mat is None:
            cov_mat = self.cov_mat
        if comp_mat is None:
            comp_mat = self.comp_mat
        target_func = lambda gamma: self.make_target_function(gamma, cov_mat=cov_mat)
        sol = spo.minimize(fun=target_func, x0=np.ones(len(self.unrestricted)))
        rotation = np.linalg.inv(self.assign_non_zero_elements(sol.x))
        return rotation
