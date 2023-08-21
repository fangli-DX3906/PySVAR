import numpy as np
from pandas import date_range
import datetime
from typing import Literal
import random

from estimation.Estimation import Estimation


class BaseModel(Estimation):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 date_start: datetime.datetime,
                 date_end: datetime.datetime,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data, constant)
        self.n_obs, self.n_vars = self.data.shape
        self.var_names = var_names
        if self.n_vars != len(self.var_names):
            raise ValueError('Names are not consistent with data dimension!')
        self.date_start = date_start
        self.date_end = date_end
        self.date_time_span = list(date_range(start=date_start, end=date_end, freq=date_frequency).to_pydatetime())
        self.date_frequency = date_frequency
        self.lag_order = self.optim_lag(y=self.data, criterion=info_criterion)
        self.identity = np.eye(self.n_vars)
        self.constant = constant
        self.criterion = info_criterion

    def fit(self) -> None:
        self.comp_mat, self.cov, self.res, self.interp, self.x = self.estimate(self.data, self.lag_order)
        self.coeff_mat = self.comp_mat[:self.n_vars]
        self.cov_mat = self.cov[:self.n_vars, :self.n_vars]
        self.intercepts = self.interp[:self.n_vars]
        zs = np.zeros((self.lag_order, self.n_vars))
        self.resids = np.concatenate((zs, self.res[:self.n_vars, :].T), axis=0)  # this is the true residuals
        self.ar_coeff = dict()
        for i in range(0, self.lag_order):
            self.ar_coeff[str(i + 1)] = self.comp_mat[:self.n_vars, i * self.n_vars:(i + 1) * self.n_vars]
        self.prepare_bootstrap()
        self.pack_likelihood_info()

    def pack_likelihood_info(self) -> None:
        Bhat = np.column_stack((self.intercepts, self.comp_mat[:self.n_vars, :]))
        Bhat = Bhat.T
        self.likelihood_info = {'Y': self.data, 'X': self.x.T, 'Bhat': Bhat, 'sigma': self.cov_mat,
                                'n': self.n_vars, 't': self.n_obs, 'p': self.lag_order}

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
