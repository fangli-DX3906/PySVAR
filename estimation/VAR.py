from typing import Union, Literal, List, Optional
import numpy as np
import random

from Base import BaseModel
from utils.Tools import Tools
from utils.Plotter import Plotter


class VAR(BaseModel):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 date_frequency: Literal['M', 'Q', 'A']= None,
                 date_start: Optional[str] = None,
                 lag_order: Optional[int] = None,
                 shock_names: Optional[List[str]] = None,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data=data, var_names=var_names, date_frequency=date_frequency, date_start=date_start,
                         lag_order=lag_order, constant=constant, info_criterion=info_criterion)

        if shock_names is not None:
            self.shock_names = shock_names
        else:
            self.shock_names = [f'Shock{_ + 1}' for _ in range(self.n_vars)]

        self.plotter = Plotter(var_names=var_names, shock_names=self.shock_names, date_frequency=date_frequency)
        self.H = self.n_obs - self.lag_order

    def solve(self):
        self.fit()
        self.tool = Tools(data=self.data,
                          lag_order=self.lag_order,
                          comp_mat=self.comp_mat,
                          cov_mat=self.cov_mat,
                          rotation=np.eye(self.n_vars))
        self.irf_point_estimate = self.tool.var_irf_point_estimate
        self.vd_point_estimate = self.tool.estimate_vd(self.tool.var_irf_point_estimate)

    def irf(self, h: int) -> np.ndarray:
        # irf_point_estimate keeps track of the longest possible IRF, so does vd_point_estimate
        return self.irf_point_estimate[:, :h + 1]

    def vd(self, h: int) -> np.ndarray:
        return self.vd_point_estimate[:, :h + 1]

    def bootstrap(self,
                  n_path: int = 100,
                  seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.irf_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H + 1))
        self.vd_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H + 1))

        for r in range(n_path):
            yr = self.make_bootstrap_sample()
            comp_mat_r, cov_mat_r, _, _, _ = self.estimate(yr, self.lag_order)
            cov_mat_r = cov_mat_r[:self.n_vars, :self.n_vars]
            self.tool.update(data=yr, comp=comp_mat_r, cov=cov_mat_r)
            self.tool.estimate_irf()
            self.irf_mat_full[r, :, :] = self.tool.irf
            self.vd_mat_full[r, :, :] = self.tool.estimate_vd(self.tool.irf)

    def calc_confid_intvl(self,
                          h: int,
                          which: Literal['irf', 'vd'],
                          sigs: Union[List[int], int]) -> dict:
        if which == 'irf':
            if 'irf_mat_full' not in self.__dir__():
                raise ValueError("bootstrap first")
            mat = self.irf_mat_full[:, :, :h + 1]
        else:
            if 'vd_mat_full' not in self.__dir__():
                raise ValueError("bootstrap first")
            mat = self.vd_mat_full[:, :, :h + 1]
        return self.tool.make_confid_intvl(mat=mat, sigs=sigs)

    def plot_irf(self,
                 h: int,
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
                raise ValueError('Not specifying the significance levels.')
            if not isinstance(sigs, list):
                sigs = [sigs]
            cv_plot = self.calc_confid_intvl(h=h, which='irf', sigs=sigs)
        else:
            cv_plot = None

        irf_plot = self.irf_point_estimate[:, :h + 1]

        if var_list is None:
            var_list = self.var_names
        elif not set(var_list).issubset(set(self.var_names)):
            raise ValueError('Check the variable names!')
        else:
            pass

        if shock_list is None:
            shock_list = self.shock_names
        elif not set(shock_list).issubset(set(range(self.n_vars))):
            raise ValueError('Check the shock names!')
        else:
            pass

        self.plotter.plot_irf(h=h + 1,
                              var_list=var_list,
                              shock_list=shock_list,
                              sigs=sigs,
                              irf=irf_plot,
                              with_ci=with_ci,
                              max_cols=max_cols,
                              irf_cv=cv_plot,
                              save_path=save_path)

    def plot_vd(self,
                h: int,
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
        else:
            pass

        if shock_list is None:
            shock_list = self.shock_names
        elif not set(shock_list).issubset(set(range(self.n_vars))):
            raise ValueError(f'The system only allows {self.n_vars} orthogonal shocks!')
        else:
            pass

        self.plotter.plot_vd(h=h + 1,
                             var_list=var_list,
                             shock_list=shock_list,
                             vd=self.vd_point_estimate,
                             max_cols=max_cols,
                             save_path=save_path)
