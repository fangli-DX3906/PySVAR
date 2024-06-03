import random
import numpy as np
from tqdm import tqdm
from typing import Union, Literal, Optional

from base_model import Model
from auxillary.bricks import estim_sys


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
                         shock_names=shock_names,
                         constant=constant,
                         lag_order=lag_order,
                         max_lag_order=max_lag_order,
                         info_criterion=info_criterion,
                         date_frequency=date_frequency,
                         date_start=date_start)

        self.fit()
        self.chol = np.linalg.cholesky(self.cov_mat)

    def get_structural_shocks(self,
                              chol: Optional[np.ndarray],
                              rotation: Optional[np.ndarray],
                              reduced_residual: Optional[np.ndarray]) -> np.ndarray:
        if chol is None:
            chol = self.chol
        if rotation is None:
            rotation = self.rotation
        if reduced_residual is None:
            reduced_residual = self.residuals
        shocks = np.dot(np.linalg.inv(np.dot(chol, rotation)), reduced_residual)

        return shocks


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

    def calc_point_estimate(self, how: Literal['median', 'average'] = 'median') -> None:
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

        if how == 'median':
            self.irf_point_estimate = np.percentile(self.irf_mat_full, 50, axis=0)
            self.vd_point_estimate = np.percentile(self.vd_mat_full, 50, axis=0)
            self.hd_point_estimate = np.percentile(self.hd_mat_full, 50, axis=0)
        else:
            self.irf_point_estimate = np.sum(self.irf_mat_full, axis=0) / self.irf_mat_full.shape[0]
            self.vd_point_estimate = np.sum(self.vd_mat_full, axis=0) / self.vd_mat_full.shape[0]
            self.hd_point_estimate = np.sum(self.hd_mat_full, axis=0) / self.hd_mat_full.shape[0]


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
                  n_rotation: int = 100,
                  seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.irf_mat_full = np.zeros((n_rotation, self.n_vars ** 2, self.H + 1))
        self.vd_mat_full = np.zeros((n_rotation, self.n_vars ** 2, self.H + 1))
        self.hd_mat_full = np.zeros((n_rotation, self.n_vars ** 2, self.H))
        self.rotation_list = []

        for r in tqdm(range(n_rotation), desc=f'Drawing {n_rotation} rotations...'):
            yr = self.make_bootstrap_sample()
            comp_mat_r, cov_mat_r, resid_r, _, _ = estim_sys(data=yr,
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
