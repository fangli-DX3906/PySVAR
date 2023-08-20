import datetime
import random
from typing import Union, Literal, List, Optional
import numpy as np

from ReducedModel import ReducedModel


class VAR(ReducedModel):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 date_start: datetime.datetime,
                 date_end: datetime.datetime,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data=data,
                         var_names=var_names,
                         date_frequency=date_frequency,
                         date_start=date_start,
                         date_end=date_end,
                         constant=constant,
                         info_criterion=info_criterion)
        self.H = self.n_obs - self.lag_order

    def irf(self, h: int) -> np.ndarray:
        self.irf_max_point_estimate = self._ReducedModel__get_irf(h=self.H, comp_mat=self.comp_mat,
                                                                  cov_mat=self.cov_mat)
        self.irf_point_estimate = self.irf_max_point_estimate[:, :h + 1]
        return self.irf_point_estimate

    def vd(self, h: int) -> np.ndarray:
        self.vd_point_estimate = self._ReducedModel__get_vd(self.irf_max_point_estimate[:, :h + 1])
        return self.vd_point_estimate

    def bootstrap(self,
                  h: int,
                  n_path: int = 100,
                  seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.irf_mat = np.zeros((n_path, self.n_vars ** 2, h + 1))
        self.vd_mat = np.zeros((n_path, self.n_vars ** 2, h + 1))
        self.irf_max_mat = np.zeros((n_path, self.n_vars ** 2, self.H + 1))

        for r in range(n_path):
            yr = self._ReducedModel__make_bootstrap_sample()
            comp_mat_r, cov_mat_r, _, _, _ = self._Estimation__estimate(yr, self.lag_order)
            cov_mat_r = cov_mat_r[:self.n_vars, :self.n_vars]
            # TODO: next version, historical decomposition allows confidence interval
            irfr = self._ReducedModel__get_irf(h=self.H, comp_mat=comp_mat_r, cov_mat=cov_mat_r)
            self.irf_max_mat[r, :, :] = irfr
            _irfr = irfr[:, :h + 1]
            self.irf_mat[r, :, :] = _irfr
            vdr = self._ReducedModel__get_vd(irfs=_irfr)
            self.vd_mat[r, :, :] = vdr

    def irf_cv(self, sig_irf: Union[List[int], int]) -> None:
        if 'irf_mat' not in self.__dir__():
            raise ValueError("bootstrap first")
        self.irf_confid_intvl = self._ReducedModel__make_confid_intvl(mat=self.irf_mat, sigs=sig_irf)

    def vd_cv(self, sig_vd: Union[List[int], int]) -> None:
        if 'vd_mat' not in self.__dir__():
            raise ValueError("bootstrap first")
        self.vd_confid_intvl = self._ReducedModel__make_confid_intvl(mat=self.vd_mat, sigs=sig_vd)

    def plot_irf(self,
                 var_list: Optional[List[str]] = None,
                 shock_list: Union[List[int]] = None,
                 sigs: Union[List[int], int] = None,
                 max_cols: int = 3,
                 with_ci: bool = True,
                 save_path: Optional[str] = None) -> None:
        if 'irf_point_estimate' not in self.__dir__():
            raise ValueError("IRFs should be estimated.")

        if with_ci:
            if sigs is None:
                raise ValueError('Not specifying significance levels.')
            if not isinstance(sigs, list):
                sigs = [sigs]
            if 'irf_confid_intvl' not in self.__dir__():
                self.irf_cv(sigs)

        if self.irf_point_estimate.shape[1] != self.irf_mat.shape[2]:
            print('Warning: length for point estimate and confidence interval are not consistent!')
            h = min(self.irf_point_estimate.shape[1], self.irf_mat.shape[2])
        else:
            h = self.irf_point_estimate.shape[1]

        if var_list is None:
            var_list = self.var_names
        elif not set(var_list).issubset(set(self.var_names)):
            raise ValueError('Check the variable names!')

        if shock_list is None:
            shock_list = self.shock_names
        elif not set(shock_list).issubset(set(range(self.n_vars))):
            raise ValueError(f'The system only allows {self.n_vars} orthogonal shocks!')
        else:
            _shock_list = []
            for i in shock_list:
                _shock_list.append(f'orth_shock_{i + 1}')
            shock_list = _shock_list

        self._ReducedModel__make_irf_graph(h=h, var_list=var_list, shock_list=shock_list,
                                           sigs=sigs, max_cols=max_cols, with_ci=with_ci)

    def plot_vd(self,
                var_list: Optional[List[str]] = None,
                shock_list: Optional[List[int]] = None,
                max_cols: int = 3,
                save_path: Optional[str] = None) -> None:
        if 'vd_point_estimate' not in self.__dir__():
            raise ValueError("IRFs should be estimated.")

        if var_list is None:
            var_list = self.var_names
        elif not set(var_list).issubset(set(self.var_names)):
            raise ValueError('Check the variable names!')

        if shock_list is None:
            shock_list = self.shock_names
        elif not set(shock_list).issubset(set(range(self.n_vars))):
            raise ValueError(f'The system only allows {self.n_vars} orthogonal shocks!')
        else:
            _shock_list = []
            for i in shock_list:
                _shock_list.append(f'orth_shock_{i + 1}')
            shock_list = _shock_list

        h = self.vd_point_estimate.shape[1]
        self._ReducedModel__make_vd_graph(h=h, var_list=var_list, shock_list=shock_list, max_cols=max_cols,
                                          save_path=save_path)
