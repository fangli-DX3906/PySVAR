import datetime
import random
from typing import Union, Literal, List, Optional
import numpy as np

from Base import BaseModel
from Plotter import Plotter
from Tools import Tools


class VAR(BaseModel):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 date_start: datetime.datetime,
                 date_end: datetime.datetime,
                 lag_order: Optional[int] = None,
                 shock_names: Optional[List[str]] = None,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data=data,
                         var_names=var_names,
                         date_frequency=date_frequency,
                         date_start=date_start,
                         date_end=date_end,
                         lag_order=lag_order,
                         constant=constant,
                         info_criterion=info_criterion)
        if shock_names is not None:
            self.shock_names = shock_names
        else:
            self.shock_names = [f'shock{_ + 1}' for _ in range(self.n_vars)]
        self.fit()
        self.tool = Tools(data=data,
                          lag_order=self.lag_order,
                          comp_mat=self.comp_mat,
                          cov_mat=self.cov_mat,
                          rotation=np.eye(self.n_vars))
        self.plotter = Plotter(var_names=var_names,
                               shock_names=self.shock_names,
                               date_frequency=date_frequency)
        self.H = self.n_obs - self.lag_order

    def irf(self, h: int) -> np.ndarray:
        self.irf_point_estimate = self.tool._irfs_
        return self.irf_point_estimate[:, :h + 1]

    def vd(self, h: int) -> np.ndarray:
        self.vd_point_estimate = self.tool.estimate_vd(self.tool._irfs_)
        return self.vd_point_estimate[:, :h + 1]

    def bootstrap(self,
                  h: int,
                  n_path: int = 100,
                  seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        # self.irf_mat = np.zeros((n_path, self.n_vars ** 2, h + 1))
        # self.vd_mat = np.zeros((n_path, self.n_vars ** 2, h + 1))
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
            # irfr = self.tool.irf
            # self.irf_mat_full[r, :, :] = irfr
            # temp_irfr = irfr[:, :h + 1]
            # self.irf_mat[r, :, :] = temp_irfr
            # vdr = self.tool.estimate_vd(temp_irfr)
            # self.vd_mat[r, :, :] = vdr

    def irf_cv(self,
               h: int,
               sigs: Union[List[int], int]) -> dict:
        if 'irf_mat_full' not in self.__dir__():
            raise ValueError("bootstrap first")
        self.irf_confid_intvl = self.tool.make_confid_intvl(mat=self.irf_mat_full[:, :h + 1], sigs=sigs)
        return self.irf_confid_intvl

    def vd_cv(self,
              h: int,
              sigs: Union[List[int], int]) -> dict:
        if 'vd_mat_full' not in self.__dir__():
            raise ValueError("bootstrap first")
        self.vd_confid_intvl = self.tool.make_confid_intvl(mat=self.vd_mat_full[:, :h + 1], sigs=sigs)
        return self.vd_confid_intvl

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
                raise ValueError('Not specifying significance levels.')
            if not isinstance(sigs, list):
                sigs = [sigs]
            cv_plot = self.irf_cv(h=h, sigs=sigs)
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
