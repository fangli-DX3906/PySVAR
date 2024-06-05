import numpy as np
import scipy.optimize as spo
from scipy.linalg import null_space
from typing import Literal, Optional

from estimation.svar import PointIdentifiedSVAR


class OptimIdentification(PointIdentifiedSVAR):
    """

    Should implement the following two types of method:
        - target_function(self, gamma, comp_mat: np.ndarray, cov_mat: np.ndarray)
        - constraint functions with the signature: constraint_name_type(self, gamma)
            name part indicates the name of the constraints
            type part indicates the type of the constraints, should be either 'eq' or 'ineq'

    Note:
        1. equality constraint means that the constraint function result is to be zero whereas
           inequality means that it is to be non-negative.
        2. could have several constraint functions

    """

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

    def parse_cons_funcs(self) -> tuple:
        cons_list = []
        for fun in self.__dir__():
            if 'constraint' in fun:
                segs = fun.split('_')
                cons_list.append({'type': segs[-1], 'fun': getattr(self, fun)})

        return tuple(cons_list)

    def solve(self,
              comp_mat: Optional[np.ndarray] = None,
              cov_mat: Optional[np.ndarray] = None) -> Optional[np.ndarray]:

        if comp_mat is None:
            comp_mat = self.comp_mat
        if cov_mat is None:
            cov_mat = self.cov_mat

        target_func = lambda gamma: self.target_function(gamma, comp_mat=comp_mat, cov_mat=cov_mat)
        cons = self.parse_cons_funcs()

        kwargs = {'fun': target_func, 'constraints': cons, 'x0': np.ndarray(self.n_vars)}
        res = spo.minimize(**kwargs)

        if res.success:
            gam = res.x.reshape((1, -1))
            rotation = np.concatenate((gam.T, null_space(gam)), axis=1)
            return rotation
        else:
            print(res.message)
            return None
