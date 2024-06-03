import random
import numpy as np
from typing import Literal, Optional, List, Union

from auxillary.bricks import estim_sys, optim_lag
from auxillary.plotting import Plotting
from auxillary.tools import Tools
from utils.date_parser import DateParser


class Model:
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

        self.data = data
        self.n_obs, self.n_vars = self.data.shape
        self.var_names = var_names
        self.date_frequency = date_frequency
        self.identity = np.eye(self.n_vars)
        self.constant = constant

        # address the var name issue
        if len(self.var_names) < self.n_vars:
            raise ValueError('missing variable names')

        # assign the shock names
        if shock_names is not None:
            self.shock_names = shock_names
        else:
            self.shock_names = [f'reduced shock {_ + 1}' for _ in range(self.n_vars)]
        self.n_shocks = len(self.shock_names)

        # select the optimal lag order
        if lag_order:
            self.lag_order = lag_order
        else:
            self.lag_order = optim_lag(data=self.data, criterion=info_criterion,
                                       max_lags=max_lag_order, constant=self.constant)
        self.H = self.n_obs - self.lag_order

        # make date list for historical decomposition
        if date_start is not None:
            self.dates = DateParser(start=date_start, n_dates=self.n_obs, fequency=date_frequency)
            self.hd_dates = self.dates.date_list[self.lag_order:]
        else:
            self.dates = list(range(1, self.n_obs + 1))
            self.hd_dates = self.dates[self.lag_order:]

        # initialize a plotting instance
        self.plots = Plotting(var_names=var_names,
                              shock_names=self.shock_names,
                              date_frequency=date_frequency)

    def fit(self) -> None:
        self.comp_mat, self.cov, self.res, self.interp, self.x = estim_sys(data=self.data,
                                                                           lag=self.lag_order,
                                                                           constant=self.constant)
        self.coeff_mat = self.comp_mat[:self.n_vars, :]
        self.cov_mat = self.cov[:self.n_vars, :self.n_vars]
        if self.constant:
            self.intercepts = self.interp[:self.n_vars]
        self.residuals = self.res[:self.n_vars, :]
        self.ar_coeff = dict()
        for i in range(0, self.lag_order):
            self.ar_coeff[str(i + 1)] = self.comp_mat[:self.n_vars, i * self.n_vars:(i + 1) * self.n_vars]

        self.tools = Tools(data=self.data,
                           lag_order=self.lag_order,
                           comp_mat=self.comp_mat,
                           cov_mat=self.cov_mat,
                           rotation=self.identity)
        self.prepare_bootstrap()
        self.pack_likelihood_info()

    def pack_likelihood_info(self) -> None:
        if self.constant:
            Bhat = np.column_stack((self.intercepts, self.comp_mat[:self.n_vars, :]))
        else:
            Bhat = self.comp_mat[:self.n_vars, :]
        Bhat = Bhat.T
        self.likelihood_info = {'Y': self.data, 'X': self.x.T, 'Bhat': Bhat,
                                'sigma': self.cov_mat, 'n': self.n_vars, 't': self.n_obs,
                                'p': self.lag_order, 'const': self.constant}

    def prepare_bootstrap(self) -> None:
        self._data_T = self.data.T
        self._yy = self._data_T[:, self.lag_order - 1:self.n_obs]

        for i in range(1, self.lag_order):
            self._yy = np.concatenate((self._yy, self._data_T[:, self.lag_order - i - 1:self.n_obs - i]), axis=0)

        self._yyr = np.zeros((self.lag_order * self.n_vars, self.n_obs - self.lag_order + 1))
        self._index_set = range(self.n_obs - self.lag_order)

    def make_bootstrap_sample(self) -> np.ndarray:
        pos = random.randint(0, self.n_obs - self.lag_order)
        self._yyr[:, 0] = self._yy[:, pos]
        idx = np.random.choice(self._index_set, size=self.n_obs - self.lag_order)
        ur = np.concatenate((np.zeros((self.lag_order * self.n_vars, 1)), self.res[:, idx]), axis=1)

        for i in range(1, self.n_obs - self.lag_order + 1):
            self._yyr[:, i] = self.interp.T + np.dot(self.comp_mat, self._yyr[:, i - 1]) + ur[:, i]

        yr = self._yyr[:self.n_vars, :]
        for i in range(1, self.lag_order):
            temp = self._yyr[i * self.n_vars:(i + 1) * self.n_vars, 0].reshape((-1, 1))
            yr = np.concatenate((temp, yr), axis=1)

        yr = yr.T

        return yr

    def irf(self, h: int) -> np.ndarray:
        return self.irf_point_estimate[:, :h + 1]

    def vd(self, h: int) -> np.ndarray:
        return self.vd_point_estimate[:, :h + 1]

    def hd(self, h: int) -> np.ndarray:
        return self.hd_point_estimate[:, :h + 1]

    def plot_irf(self,
                 h: int,
                 var_list: Optional[List[str]] = None,
                 shock_list: Optional[List[str]] = None,
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
                shock_list: Optional[List[str]] = None,
                save_path: Optional[str] = None) -> None:

        if 'vd_point_estimate' not in self.__dir__():
            raise ValueError('VDs should be estimated.')

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

    def plot_hd(self) -> None:
        pass
