import random
import numpy as np
from typing import Union, Literal, List, Optional

from base_model import Model
from auxillary.tools import Tools
from auxillary.plotting import Plotting
from auxillary.bricks import estimate_var


class VAR(Model):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: Optional[List[str]] = None,
                 constant: bool = True,
                 lag_order: Optional[int] = None,
                 max_lag_order: Optional[int] = 8,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 date_frequency: Literal['M', 'Q', 'A'] = 'Q',
                 date_start: str = None):

        super().__init__(data=data,
                         var_names=var_names,
                         constant=constant,
                         lag_order=lag_order,
                         max_lag_order=max_lag_order,
                         info_criterion=info_criterion,
                         date_frequency=date_frequency,
                         date_start=date_start)

        if shock_names is not None:
            self.shock_names = shock_names
        else:
            self.shock_names = [f'reduced shock {_ + 1}' for _ in range(self.n_vars)]

        self.plots = Plotting(var_names=var_names,
                              shock_names=self.shock_names,
                              date_frequency=date_frequency)

        self.H = self.n_obs - self.lag_order
        self.hd_dates = self.dates.date_list[self.lag_order:]

    def solve(self):
        self.fit()
        self.tools = Tools(data=self.data,
                           lag_order=self.lag_order,
                           comp_mat=self.comp_mat,
                           cov_mat=self.cov_mat,
                           rotation=self.identity)

        self.irf_point_estimate = self.tools.reduced_var_irf_point_estimate
        self.vd_point_estimate = self.tools.estimate_vd(self.tools.reduced_var_irf_point_estimate)
        self.hd_point_estimate = self.tools.estimate_hd(self.residuals, self.tools.reduced_var_irf_point_estimate)

    def irf(self, h: int) -> np.ndarray:
        return self.irf_point_estimate[:, :h + 1]

    def vd(self, h: int) -> np.ndarray:
        return self.vd_point_estimate[:, :h + 1]

    def hd(self, start: str, end: str) -> np.ndarray:
        start_idx = self.hd_dates.index(start)
        end_idx = self.hd_dates.index(end)
        return self.hd_point_estimate[:, start_idx: end_idx + 1]

    def bootstrap(self, n_path: int = 100, seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.irf_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H + 1))
        self.vd_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H + 1))
        self.hd_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H))

        for r in range(n_path):
            yr = self.make_bootstrap_sample()
            comp_mat_r, cov_mat_r, resid_r, _, _ = estimate_var(data=yr,
                                                                lag=self.lag_order,
                                                                constant=self.constant)
            cov_mat_r = cov_mat_r[:self.n_vars, :self.n_vars]
            residual_r = resid_r[:self.n_vars, :]
            self.tools.update(data=yr, comp=comp_mat_r, cov=cov_mat_r)
            irf_r = self.tools.estimate_irf()
            self.irf_mat_full[r, :, :] = irf_r
            self.vd_mat_full[r, :, :] = self.tools.estimate_vd(irfs=irf_r)
            self.hd_mat_full[r, :, :] = self.tools.estimate_hd(shocks=residual_r, irfs=irf_r)

    def plot_irf(self,
                 h: int,
                 var_list: Optional[List[str]] = None,
                 shock_list: Optional[List[int]] = None,
                 sigs: Union[List[int], int] = None,
                 save_path: Optional[str] = None) -> None:

        if 'irf_point_estimate' not in self.__dir__():
            raise ValueError('IRFs should be estimated.')

        if sigs is None:
            with_ci = False
        elif not isinstance(sigs, list):
            sigs = [sigs]
            with_ci = True
        else:
            with_ci = True

        if h > self.H:
            raise ValueError('length is too long.')
        cv_plot = self.tools.make_confid_intvl(mat=self.irf_mat_full, sigs=sigs, length=h + 1)
        irf_plot = self.irf_point_estimate[:, :h + 1]

        if var_list is None:
            var_list = self.var_names
        elif not set(var_list).issubset(set(self.var_names)):
            raise ValueError('variable names not valid')
        else:
            pass

        if shock_list is None:
            shock_list = self.shock_names
        elif not set(var_list).issubset(set(self.var_names)):
            raise ValueError('shock names not valid')
        else:
            pass

        self.plots.plot_irf(h=h,
                            var_list=var_list,
                            shock_list=shock_list,
                            sigs=sigs,
                            irf=irf_plot,
                            with_ci=with_ci,
                            irf_ci=cv_plot,
                            save_path=save_path)

    def plot_vd(self,
                h: int,
                var_list: Optional[List[str]] = None,
                shock_list: Optional[List[int]] = None,
                save_path: Optional[str] = None) -> None:

        if 'vd_point_estimate' not in self.__dir__():
            raise ValueError("IRFs should be estimated.")

        if h > self.H:
            raise ValueError('length is too long.')
        vd_plot = self.vd_point_estimate[:, :h + 1]

        if var_list is None:
            var_list = self.var_names
        elif not set(var_list).issubset(set(self.var_names)):
            raise ValueError('variable names not valid')
        else:
            pass

        if shock_list is None:
            shock_list = self.shock_names
        elif not set(var_list).issubset(set(self.var_names)):
            raise ValueError('shock names not valid')
        else:
            pass

        self.plots.plot_vd(h=h,
                           var_list=var_list,
                           shock_list=shock_list,
                           vd=vd_plot,
                           save_path=save_path)
