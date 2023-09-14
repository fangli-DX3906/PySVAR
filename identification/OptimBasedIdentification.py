import numpy as np
import scipy.optimize as spo
from scipy.linalg import null_space
from typing import Literal, Optional, Union
import datetime

from estimation.SVAR import PointIdentifiedSVAR

# TODO: check
class OptimBasedIdentification(PointIdentifiedSVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 target_func: Literal['irf', 'vd', 'customize'],
                 target_var: str,
                 target_how: Literal['level', 'sum'],
                 reg_var: str,
                 reg_strength: Union[float, int],
                 reg_sign: Literal[1, -1],
                 reg_how: Literal['level', 'sum'],
                 period: int,
                 pso: int,
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
        self.rotation = None
        self.target_func = target_func
        self.target_var = target_var
        self.target_how = target_how
        self.reg_var = reg_var
        self.reg_strength = reg_strength
        self.reg_sign = reg_sign
        self.reg_how = reg_how
        self.period = period
        self.pos = pso

    def update_info(self, **kwargs):
        to_be_updated = list(kwargs.keys())
        pass

    def make_target_function(self,
                             gamma,
                             comp_mat: np.ndarray,
                             cov_mat: np.ndarray):
        gamma = gamma.reshape((1, -1))
        gamma_null = null_space(gamma)
        rotation_temp = np.concatenate((gamma.T, gamma_null), axis=1)
        self.tool.update(rotation=rotation_temp)
        self.tool.estimate_irf(length=self.period)
        irf_temp = self.tool.irf
        vd_temp = self.tool.estimate_vd(irfs=irf_temp)
        idx = self.var_names.index(self.target_var)
        idx_reg = self.var_names.index(self.reg_var)
        func = irf_temp[idx, :] if self.target_func == 'irf' else vd_temp[idx, :]
        func = np.sum(func) if self.target_how == 'sum' else func[self.period]
        reg_part = irf_temp[idx_reg, :] if self.target_func == 'irf' else vd_temp[idx_reg, :]
        d = np.tile(self.reg_strength, self.period + 1)
        reg_part = np.sum(d * reg_part) if self.reg_how == 'sum' else self.reg_strength * reg_part[self.period]
        return -(func + self.reg_strength * reg_part)

    def orthogonality_constraint(self, gamma):
        return np.dot(gamma, gamma) - 1

    def other_constriant(self, gamma):
        return gamma[self.pos]

    def solve(self,
              comp_mat: Optional[np.ndarray] = None,
              cov_mat: Optional[np.ndarray] = None):
        if comp_mat is None:
            comp_mat = self.comp_mat
        if cov_mat is None:
            cov_mat = self.cov_mat

        target_func = lambda gamma: self.make_target_function(gamma, comp_mat=comp_mat, cov_mat=cov_mat)
        cons = ({'type': 'eq', 'fun': self.orthogonality_constraint}, {'type': 'eq', 'fun': self.other_constriant})
        sol = spo.minimize(fun=target_func, x0=np.ones(self.n_vars), constraints=cons)
        gam = sol.x.reshape((1, -1))
        rotation = np.concatenate((gam.T, null_space(gam)), axis=1)
        return rotation
