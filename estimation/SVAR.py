import datetime
import random
from typing import Union, Literal, List, Optional
import numpy as np

from Base import BaseModel
from Plotter import Plotter
from Tools import Tools


class SVAR(BaseModel):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 date_start: Optional[datetime.datetime],
                 date_end: Optional[datetime.datetime],
                 lag_order: Optional[int] = None,
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
        self.shock_names = shock_names
        self.H = self.n_obs - self.lag_order
        self.n_shocks = len(shock_names)
        self.fit()
        self.chol = np.linalg.cholesky(self.cov_mat)
        self.plotter = Plotter(var_names=var_names,
                               shock_names=self.shock_names,
                               date_frequency=date_frequency)
        self.tool = Tools(data=data,
                          lag_order=self.lag_order,
                          comp_mat=self.comp_mat,
                          cov_mat=self.cov_mat)

    def get_structural_shocks(self,
                              chol: Union[np.ndarray, None],
                              rotation: Union[np.ndarray, None],
                              resid: Union[np.ndarray, None]) -> np.ndarray:
        if chol is None:
            chol = self.chol
        if rotation is None:
            rotation = self.rotation
        if resid is None:
            resid = self.resids
        shocks = np.dot(np.linalg.inv(np.dot(chol, rotation)), resid[self.lag_order:, :].T)
        return shocks

    def plot_irf(self,
                 h: int,
                 sigs: Union[List[int], int] = None,
                 var_list: Optional[List[str]] = None,
                 shock_list: Optional[List[str]] = None,
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
        elif not set(shock_list).issubset(set(self.shock_names)):
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

        vd_plot = self.vd_point_estimate[:, :h + 1]
        self.plotter.plot_vd(h=h + 1,
                             var_list=var_list,
                             shock_list=shock_list,
                             vd=vd_plot,
                             max_cols=max_cols,
                             save_path=save_path)

    def calc_confid_intvl(self,
                          h: int,
                          which: Literal['irf', 'vd'],
                          sigs: Union[List[int], int]) -> dict:
        if which == 'irf':
            if 'irf_mat_full' not in self.__dir__():
                raise ValueError("bootstrap first")
            mat = self.irf_mat_full[:, :self.n_shocks * self.n_vars, :h + 1]
        else:
            if 'vd_mat_full' not in self.__dir__():
                raise ValueError("bootstrap first")
            mat = self.vd_mat_full[:, :self.n_shocks * self.n_vars, :h + 1]
        return self.tool.make_confid_intvl(mat=mat, sigs=sigs)


class SetIdentifiedSVAR(SVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 date_start: Optional[datetime.datetime],
                 date_end: Optional[datetime.datetime],
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

    def full_irf(self) -> None:
        n_rotation = len(self.rotation_list)
        self.irf_mat_full = np.zeros((n_rotation, self.n_vars ** 2, self.H + 1))
        self.vd_mat_full = np.zeros((n_rotation, self.n_vars ** 2, self.H + 1))
        counter = 1
        for rotation in self.rotation_list:
            self.tool.update(rotation=rotation)
            self.tool.estimate_irf()
            self.irf_mat_full[counter - 1, :, :] = self.tool.irf
            self.vd_mat_full[counter - 1, :, :] = self.tool.estimate_vd(irfs=self.tool.irf)
            counter += 1

    def irf(self,
            h: int,
            how: Literal['median', 'average'] = 'median') -> np.ndarray:
        if 'irf_mat_full' not in self.__dir__():
            raise ValueError("Model is not identified.")
        if how == 'median':
            self.irf_point_estimate = np.percentile(self.irf_mat_full, 50, axis=0)
        elif how == 'average':
            self.irf_point_estimate = np.sum(self.irf_mat_full, axis=0) / self.irf_mat_full.shape[0]
        return self.irf_point_estimate[:self.n_shocks * self.n_vars, :h + 1]

    def vd(self,
           h: int,
           how: Literal['median', 'average'] = 'median') -> np.ndarray:
        if 'vd_mat_full' not in self.__dir__():
            raise ValueError("Model is not identified.")
        if how == 'median':
            self.vd_point_estimate = np.percentile(self.vd_mat_full, 50, axis=0)
        elif how == 'average':
            self.vd_point_estimate = np.sum(self.vd_mat_full, axis=0) / self.vd_mat_full.shape[0]
        return self.vd_point_estimate[:self.n_shocks * self.n_vars, :h + 1]


class PointIdentifiedSVAR(SVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 date_start: Optional[datetime.datetime],
                 date_end: Optional[datetime.datetime],
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

    def identify(self):
        self.rotation = self.solve()
        self.tool.update(rotation=self.rotation)
        self.tool.estimate_irf()
        self.irf_point_estimate = self.tool.irf
        self.vd_point_estimate = self.tool.estimate_vd(irfs=self.tool.irf)
        return self.rotation

    def bootstrap(self,
                  n_path: int = 100,
                  seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.irf_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H + 1))
        self.vd_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H + 1))
        self.rotation_list = []

        for r in range(n_path):
            yr = self.make_bootstrap_sample()
            comp_mat_r, cov_mat_r, res_r, _, _ = self.estimate(yr, self.lag_order)
            cov_mat_r = cov_mat_r[:self.n_vars, :self.n_vars]
            rotationr = self.solve(comp_mat=comp_mat_r, cov_mat=cov_mat_r)
            self.rotation_list.append(rotationr)
            self.tool.update(data=yr, comp=comp_mat_r, cov=cov_mat_r, rotation=rotationr)
            self.tool.estimate_irf()
            self.irf_mat_full[r, :, :] = self.tool.irf
            self.vd_mat_full[r, :, :] = self.tool.estimate_vd(self.tool.irf)

    def irf(self, h: int) -> np.ndarray:
        return self.irf_point_estimate[:, :h + 1]

    def vd(self, h: int) -> np.ndarray:
        return self.vd_point_estimate[:, :h + 1]
