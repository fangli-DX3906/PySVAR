import random
import numpy as np
from typing import Union, Literal, List, Optional

from base_model import Model
from auxillary.plotting import Plotting
from auxillary.tools import Tools
from auxillary.bricks import estimate_var


class SVAR(Model):
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
                         constant=constant,
                         lag_order=lag_order,
                         max_lag_order=max_lag_order,
                         info_criterion=info_criterion,
                         date_frequency=date_frequency,
                         date_start=date_start)

        self.shock_names = shock_names
        self.H = self.n_obs - self.lag_order
        self.n_shocks = len(shock_names)
        self.fit()

        self.chol = np.linalg.cholesky(self.cov_mat)
        self.plots = Plotting(var_names=var_names,
                              shock_names=self.shock_names,
                              date_frequency=date_frequency)

        self.tools = Tools(data=data,
                           lag_order=self.lag_order,
                           comp_mat=self.comp_mat,
                           cov_mat=self.cov_mat)

    def get_structural_shocks(self,
                              chol: Union[np.ndarray, None],
                              rotation: Union[np.ndarray, None],
                              reduced_residual: Union[np.ndarray, None]) -> np.ndarray:
        if chol is None:
            chol = self.chol
        if rotation is None:
            rotation = self.rotation
        if reduced_residual is None:
            reduced_residual = self.residuals
        shocks = np.dot(np.linalg.inv(np.dot(chol, rotation)), reduced_residual)

        return shocks

    def plot_irf(self,
                 h: int,
                 var_list: Optional[List[str]] = None,
                 shock_list: Optional[List[int]] = None,
                 sigs: Union[List[int], int] = None,
                 save_path: Optional[str] = None) -> None:

        if 'irf_point_estimate' not in self.__dir__():
            raise ValueError("IRFs should be estimated.")

        if sigs is None:
            with_ci = False
        elif not isinstance(sigs, list):
            sigs = [sigs]
            with_ci = True
        else:
            with_ci = True

        if h > self.H:
            raise ValueError('length is too long.')
        irf_plot = self.irf_point_estimate[:, :h + 1]
        cv_plot = self.tools.make_confid_intvl(mat=self.irf_mat_full, sigs=sigs, length=h + 1)

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
            raise ValueError('Check the variable names!')
        else:
            pass

        if shock_list is None:
            shock_list = self.shock_names
        elif not set(shock_list).issubset(set(range(self.n_vars))):
            raise ValueError(f'The system only allows {self.n_vars} orthogonal shocks!')
        else:
            pass

        self.plots.plot_vd(h=h,
                           var_list=var_list,
                           shock_list=shock_list,
                           vd=vd_plot,
                           save_path=save_path)


class SetIdentifiedSVAR(SVAR):
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

    def _full_irf(self) -> None:
        n_rotation = len(self.rotation_list)
        self.irf_mat_full = np.zeros((n_rotation, self.n_vars ** 2, self.H + 1))
        self.vd_mat_full = np.zeros((n_rotation, self.n_vars ** 2, self.H + 1))
        self.hd_mat_full = np.zeros((n_rotation, self.n_vars ** 2, self.H))

        counter = 1
        for rotation in self.rotation_list:
            self.tools.update(rotation=rotation)
            irf_r = self.tools.estimate_irf()
            self.irf_mat_full[counter - 1, :, :] = irf_r
            self.vd_mat_full[counter - 1, :, :] = self.tools.estimate_vd(irfs=irf_r)
            shock_r = self.get_structural_shocks(chol=self.chol,
                                                 rotation=rotation,
                                                 reduced_residual=self.residuals)
            self.hd_mat_full[counter - 1, :, :] = self.tools.estimate_hd(shocks=shock_r, irfs=irf_r)
            counter += 1

    def irf(self, h: int, how: Literal['median', 'average'] = 'median') -> np.ndarray:
        if 'irf_mat_full' not in self.__dir__():
            raise ValueError("Model is not identified.")

        if how == 'median':
            self.irf_point_estimate = np.percentile(self.irf_mat_full, 50, axis=0)
        elif how == 'average':
            self.irf_point_estimate = np.sum(self.irf_mat_full, axis=0) / self.irf_mat_full.shape[0]

        return self.irf_point_estimate[:self.n_shocks * self.n_vars, :h + 1]

    def vd(self, h: int, how: Literal['median', 'average'] = 'median') -> np.ndarray:
        if 'vd_mat_full' not in self.__dir__():
            raise ValueError("Model is not identified.")

        if how == 'median':
            self.vd_point_estimate = np.percentile(self.vd_mat_full, 50, axis=0)
        elif how == 'average':
            self.vd_point_estimate = np.sum(self.vd_mat_full, axis=0) / self.vd_mat_full.shape[0]

        return self.vd_point_estimate[:self.n_shocks * self.n_vars, :h + 1]

    def hd(self, h: int, how: Literal['median', 'average'] = 'median') -> np.ndarray:
        if 'hd_mat_full' not in self.__dir__():
            raise ValueError("Model is not identified.")

        if how == 'median':
            self.hd_point_estimate = np.percentile(self.hd_mat_full, 50, axis=0)
        elif how == 'average':
            self.hd_point_estimate = np.sum(self.hd_mat_full, axis=0) / self.hd_mat_full.shape[0]

        return self.hd_point_estimate[:self.n_shocks * self.n_vars, :h + 1]


class PointIdentifiedSVAR(SVAR):
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

    def identify(self):
        self.rotation = self.solve()
        self.tools.update(rotation=self.rotation)
        irfs = self.tools.estimate_irf()
        self.irf_point_estimate = irfs
        self.vd_point_estimate = self.tools.estimate_vd(irfs=irfs)
        self.shocks = self.get_structural_shocks(chol=self.chol, rotation=self.rotation,
                                                 reduced_residual=self.residuals)
        self.hd_point_estimate = self.tools.estimate_hd(shocks=self.shocks, irfs=irfs)
        return self.rotation

    def bootstrap(self,
                  n_path: int = 100,
                  seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.irf_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H + 1))
        self.vd_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H + 1))
        self.hd_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H))
        self.rotation_list = []

        for r in range(n_path):
            yr = self.make_bootstrap_sample()
            comp_mat_r, cov_mat_r, resid_r, _, _ = estimate_var(data=yr,
                                                                lag=self.lag_order,
                                                                constant=self.constant)
            cov_mat_r = cov_mat_r[:self.n_vars, :self.n_vars]
            rotation_r = self.solve(comp_mat=comp_mat_r, cov_mat=cov_mat_r)
            self.rotation_list.append(rotation_r)
            self.tools.update(data=yr, comp=comp_mat_r, cov=cov_mat_r, rotation=rotation_r)
            irfs_r = self.tools.estimate_irf()
            self.irf_mat_full[r, :, :] = irfs_r
            self.vd_mat_full[r, :, :] = self.tools.estimate_vd(irfs=irfs_r)
            chol_r = np.linalg.cholesky(cov_mat_r)
            resid_r = resid_r[:self.n_vars, :]
            shock_r = self.get_structural_shocks(chol=chol_r,
                                                 rotation=rotation_r,
                                                 reduced_residual=resid_r)
            self.hd_mat_full[r, :, :] = self.tools.estimate_hd(shocks=shock_r, irfs=irfs_r)

    def irf(self, h: int) -> np.ndarray:
        return self.irf_point_estimate[:, :h + 1]

    def vd(self, h: int) -> np.ndarray:
        return self.vd_point_estimate[:, :h + 1]

    def hd(self, h: int) -> np.ndarray:
        return self.hd_point_estimate[:, :h + 1]
