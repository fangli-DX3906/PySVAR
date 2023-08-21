import datetime
import random
from typing import Union, Literal, List, Optional
import numpy as np

from Base import ReducedModel


class SVAR(ReducedModel):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
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
        self.shock_names = shock_names
        self.n_shocks = len(shock_names)
        self.n_diff = self.n_vars - self.n_shocks
        self.H = self.n_obs - self.lag_order
        self.fit()
        self.chol = np.linalg.cholesky(self.cov_mat)  # cholesky gives you the lower triangle in numpy

    def get_structural_shocks(self,
                              chol: Optional[np.ndarray] = None,
                              rotation: Optional[np.ndarray] = None,
                              resid: Optional[np.ndarray] = None) -> np.ndarray:
        if chol is None:
            chol = self.chol
        if rotation is None:
            rotation = self.rotation
        if resid is None:
            resid = self.resids
        shocks = np.dot(np.linalg.inv(np.dot(chol, rotation)), resid[self.lag_order:, :].T)
        return shocks

    def irf(self, h: int) -> np.ndarray:
        return self.irf_point_estimate[:, h + 1]

    def vd(self, h: int) -> np.ndarray:
        return self.vd_point_estimate[:, h + 1]

    # TODO: check this
    def __get_hd(self,
                 shocks: np.ndarray,
                 irfs: np.ndarray) -> np.ndarray:
        hd = np.zeros((self.n_vars, self.n_obs - self.lag_order, self.n_vars))
        for iperiod in range(self.n_obs - self.lag_order):
            for ishock in range(self.n_vars):
                for ivar in range(self.n_vars):
                    shocks_ = shocks[ishock, :iperiod]
                    hd[ivar, iperiod, ishock] = np.dot(irfs[ivar + ishock * self.n_vars, :iperiod], shocks_[::-1])
                    hd = hd.swapaxes(0, 2)
        return hd

    def hd(self,
           start: Optional[datetime.datetime] = None,
           end: Optional[datetime.datetime] = None) -> np.ndarray:
        if end <= start:
            raise ValueError('Invalid date!')
        if start < self.date_start:
            raise ValueError('Invalid date!')
        else:
            # TODO: check the logic
            temp_idx = list(self.date_time_span).index(start)
            start_idx = temp_idx if temp_idx >= self.lag_order else temp_idx - self.lag_order
        if end > self.date_end:
            end_idx = -1
        else:
            end_idx = list(self.date_time_span).index(end)
        return self.hd_point_estimate[:, start_idx:end_idx + 1]

    def irf_cv(self,
               irf_sig: Union[List[int], int]) -> None:
        if 'irf_mat' not in self.__dir__():
            raise ValueError("bootstrap first")
        self.irf_confid_intvl = self._ReducedModel__make_confid_intvl(mat=self.irf_mat, sigs=irf_sig)
        if self.median_as_point_estimate:
            self.irf_point_estimate = np.percentile(self.irf_mat, 50, axis=0)

    def vd_cv(self,
              vd_sig: Union[List[int], int]) -> None:
        if 'vd_mat' not in self.__dir__():
            raise ValueError("bootstrap first")
        self.vd_confid_intvl = self._ReducedModel__make_confid_intvl(mat=self.vd_mat, sigs=vd_sig)
        if self.median_as_point_estimate:
            self.vd_point_estimate = np.percentile(self.vd_mat, 50, axis=0)

    def hd_cv(self,
              hd_sig: Union[List[int], int]) -> None:
        if 'vd_mat' not in self.__dir__():
            raise ValueError("bootstrap first")
        self.vd_confid_intvl = self._ReducedModel__make_confid_intvl(mat=self.hd_mat, sigs=hd_sig)
        if self.median_as_point_estimate:
            self.vd_point_estimate = np.percentile(self.hd_mat, 50, axis=0)

    def plot_irf(self,
                 var_list: Optional[List[str]] = None,
                 shock_list: Optional[List[str]] = None,
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
        elif not set(shock_list).issubset(set(self.shock_names)):
            raise ValueError('Check the shock names!!')

        self._ReducedModel__make_irf_graph(h=h, var_list=var_list, shock_list=shock_list, sigs=sigs,
                                           max_cols=max_cols, with_ci=with_ci, save_path=save_path)

    def plot_vd(self,
                var_list: Optional[List[str]] = None,
                shock_list: Optional[List[str]] = None,
                max_cols: int = 3,
                save_path: Optional[str] = None) -> None:
        if 'vd_point_estimate' not in self.__dir__():
            raise ValueError("VD should be estimated.")

        if var_list is None:
            var_list = self.var_names
        elif not set(var_list).issubset(set(self.var_names)):
            raise ValueError('Check the variable names!')

        if shock_list is None:
            shock_list = self.shock_names
        elif not set(shock_list).issubset(set(self.shock_names)):
            raise ValueError('Check the shock names!!')

        h = self.vd_point_estimate.shape[1]
        self._ReducedModel__make_vd_graph(h=h, var_list=var_list, shock_list=shock_list,
                                          max_cols=max_cols, save_path=save_path)


class SetIdentifiedSVAR(SVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 date_start: datetime.datetime,
                 date_end: datetime.datetime,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         date_frequency=date_frequency,
                         date_start=date_start,
                         date_end=date_end,
                         constant=constant,
                         info_criterion=info_criterion)
        self.rotation_mat = None
        self.median_as_point_estimate = True


class PointIdentifiedSVAR(SVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 date_start: datetime.datetime,
                 date_end: datetime.datetime,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         date_frequency=date_frequency,
                         date_start=date_start,
                         date_end=date_end,
                         constant=constant,
                         info_criterion=info_criterion)
        self.rotation = None
        self.median_as_point_estimate = False

    def bootstrap(self,
                  h: int,
                  n_path: int,
                  seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.irf_mat = np.zeros((n_path, self.n_vars ** 2, h + 1))
        self.vd_mat = np.zeros((n_path, self.n_vars ** 2, h + 1))
        self.irf_max_mat = np.zeros((n_path, self.n_vars ** 2, self.H + 1))
        self.shock_mat = np.zeros((n_path, self.H, self.n_vars))
        self.rotation_mat = np.zeros((n_path, self.n_vars, self.n_vars))
        zs = np.zeros((self.lag_order, self.n_vars))

        for r in range(n_path):
            yr = self._ReducedModel__make_bootstrap_sample()
            comp_mat_r, cov_mat_r, res_r, _, _ = self._Estimation__estimate(yr, self.lag_order)
            cov_mat_r = cov_mat_r[:self.n_vars, :self.n_vars]
            rotationr = self.solve(comp_mat=comp_mat_r, cov_mat=cov_mat_r)
            self.rotation_mat[r, :, :] = rotationr
            irfr = self._ReducedModel__get_irf(h=self.H, rotation=rotationr, comp_mat=comp_mat_r, cov_mat=cov_mat_r)
            self.irf_max_mat[r, :, :] = irfr
            _irfr = irfr[:, :h + 1]
            self.irf_mat[r, :, :] = _irfr
            vdr = self._ReducedModel__get_vd(irfs=_irfr)
            self.vd_mat[r, :, :] = vdr
            # TODO: think about how to get HDs
            # resids_r = np.concatenate((zs, res_r[:self.n_vars, :].T), axis=0)  # this is the true residuals
            # shock_r = self.get_structural_shocks(chol=np.linalg.cholesky(cov_mat_r), rotation=rotationr, resid=resids_r)
            # self.shock_mat[r, :, :] = shock_r
            # self.hd_mat[r, :, :] = self.__get_hd(shock_r, irfr)
