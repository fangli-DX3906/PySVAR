import numpy as np
import scipy.optimize as spo
from scipy.linalg import null_space
from typing import Literal, Optional

from estimation.svar import PointIdentifiedSVAR


class OptimIdentification(PointIdentifiedSVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 constant: bool = True,
                 lag_order: Optional[int] = None,
                 max_lag_order: Optional[int] = 8,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 date_frequency: Literal['M', 'Q', 'A'] = 'Q',
                 date_start: str = None):
        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         constant=constant,
                         lag_order=lag_order,
                         max_lag_order=max_lag_order,
                         info_criterion=info_criterion,
                         date_frequency=date_frequency,
                         date_start=date_start)
        self.identification = 'optimization identification'

    # def make_target_function(self,
    #                          gamma,
    #                          comp_mat: np.ndarray,
    #                          cov_mat: np.ndarray):
    #     pass

    # def other_constraint(self,
    #                      gamma):
    #     pass

    def orthogonality_constraint(self, gamma):
        return np.dot(gamma, gamma) - 1

    def solve(self,
              comp_mat: Optional[np.ndarray] = None,
              cov_mat: Optional[np.ndarray] = None):

        if comp_mat is None:
            comp_mat = self.comp_mat
        if cov_mat is None:
            cov_mat = self.cov_mat

        target_func = lambda gamma: self.make_target_function(gamma, comp_mat=comp_mat, cov_mat=cov_mat)
        if 'other_constriant' in self.__dir__():
            cons = (
                {'type': 'eq', 'fun': self.orthogonality_constraint},
                {'type': 'eq', 'fun': self.other_constriant}
            )
        else:
            cons = (
                {'type': 'eq', 'fun': self.orthogonality_constraint},
            )

        sol = spo.minimize(fun=target_func, x0=np.ones(self.n_vars), constraints=cons)
        gam = sol.x.reshape((1, -1))
        rotation = np.concatenate((gam.T, null_space(gam)), axis=1)
        return rotation
