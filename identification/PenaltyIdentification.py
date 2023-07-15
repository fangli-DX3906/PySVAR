import numpy as np
import scipy.optimize as spo
from scipy.linalg import null_space
from typing import Union, Literal, List, Optional
import datetime

from SVAR import PointIdentifiedSVAR


class PenaltyIdentification(PointIdentifiedSVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 target_var: str,
                 target_how: Literal['level', 'sum'],
                 target_period: int,
                 target_obj: Literal['irf', 'vd'],
                 sign: Literal[1, -1],
                 reg_var: str,
                 reg_how: Literal['level', 'sum'],
                 reg_period: Optional[int] = None,
                 reg_obj: Literal['irf', 'vd'] = 'irf',
                 delta: Union[float, int] = 1e9,
                 penalty_pos: Optional[int] = None,
                 constant: bool = True,
                 data_frequency: Literal['D', 'W', 'M', 'Q', 'SA', 'A'] = 'Q',
                 date_range: List[datetime.date] = None,  # specific to HD
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data, var_names, shock_names, constant, info_criterion, data_frequency, date_range)
        self.identification = 'penalty function'
        self.target_obj = target_obj
        self.target_var = target_var
        self.target_how = target_how
        self.reg_obj = reg_obj
        self.reg_var = reg_var
        self.reg_how = reg_how
        self.target_period = target_period
        if reg_period is None:
            self.period = target_period
            self.reg_period = target_period
        else:
            self.period = max(target_period, reg_period)
            self.reg_period = reg_period
        self.delta = delta
        self.sign = sign
        self.penalty_pos = penalty_pos

    def __make_target_function(self,
                               gamma,  # this is the unknowns
                               comp_mat: np.ndarray,
                               cov_mat: np.ndarray):
        gamma = gamma.reshape((1, -1))
        gamma_null = null_space(gamma)
        rotation = np.concatenate((gamma.T, gamma_null), axis=1)
        irf = self._ReducedModel__get_irf(h=self.period, comp_mat=comp_mat, cov_mat=cov_mat, rotation=rotation)
        vd = self._ReducedModel__get_vd(irfs=irf)
        target_idx = self.var_names.index(self.target_var)
        reg_idx = self.var_names.index(self.reg_var)
        func = irf[target_idx, :] if self.target_obj == 'irf' else vd[target_idx, :]
        func = np.sum(func) if self.target_how == 'sum' else func[self.target_period]
        reg_part = irf[reg_idx, :] if self.reg_obj == 'irf' else vd[reg_idx, :]
        d = np.tile(self.delta, self.reg_period + 1)
        reg_part = np.sum(d * reg_part) if self.reg_how == 'sum' else self.delta * reg_part[self.period]
        return -(func + self.sign * reg_part)

    def __orthogonality_constraint(self, gamma):
        return np.dot(gamma, gamma) - 1

    def __zero_constriant(self, gamma):
        return gamma[self.penalty_pos]

    def solve(self,
              comp_mat: Optional[np.ndarray] = None,
              cov_mat: Optional[np.ndarray] = None):
        if comp_mat is None:
            comp_mat = self.comp_mat
        if cov_mat is None:
            cov_mat = self.cov_mat

        target_func = lambda gamma: self.__make_target_function(gamma, comp_mat=comp_mat, cov_mat=cov_mat)
        cons = ({'type': 'eq', 'fun': self.__orthogonality_constraint}, {'type': 'eq', 'fun': self.__zero_constriant})
        sol = spo.minimize(fun=target_func, x0=np.ones(self.n_vars), constraints=cons)
        gam = sol.x.reshape((1, -1))
        rotation = np.concatenate((gam.T, null_space(gam)), axis=1)

        return rotation

    def identify(self, h: int):
        if self.rotation is None:
            self.rotation = self.solve()
        self.irf_point_estimate = self._ReducedModel__get_irf(h=h, comp_mat=self.comp_mat, cov_mat=self.cov_mat,
                                                              rotation=self.rotation)
        self.vd_point_estimate = self._ReducedModel__get_vd(irfs=self.irf_point_estimate)

    def boot_confid_intvl(self,
                          h: int,
                          n_path: int,
                          irf_sig: Union[int, list],
                          vd_sig: Union[int, list, None] = None,
                          seed: Union[bool, int] = False):
        if vd_sig is None:
            vd_sig = irf_sig
        self.bootstrap(h=h, n_path=n_path, seed=seed)
        self.irf_cv(irf_mat=self.irf_mat, irf_sig=irf_sig, median_as_point_estimate=False)
        self.vd_cv(vd_mat=self.vd_mat, vd_sig=vd_sig, median_as_point_estimate=False)
