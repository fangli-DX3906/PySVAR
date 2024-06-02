import random
import numpy as np
from typing import Literal, Optional

from auxillary.bricks import estimate_var, optim_lag
from utils.date_parser import DateParser


class Model:
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
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

        # make date list for historical decomposition
        if date_start is not None:
            self.dates = DateParser(start=date_start, n_dates=self.n_obs, fequency=date_frequency)
        else:
            self.dates = list(range(1, self.n_obs + 1))

        if len(self.var_names) < self.n_vars:
            raise ValueError('missing variable names')

        if lag_order:
            self.lag_order = lag_order
        else:
            self.lag_order = optim_lag(data=self.data, criterion=info_criterion,
                                       max_lags=max_lag_order, constant=self.constant)

    def fit(self) -> None:
        self.comp_mat, self.cov, self.res, self.interp, self.x = estimate_var(data=self.data, lag=self.lag_order,
                                                                              constant=self.constant)
        self.coeff_mat = self.comp_mat[:self.n_vars, :]
        self.cov_mat = self.cov[:self.n_vars, :self.n_vars]
        self.intercepts = self.interp[:self.n_vars]
        zs = np.zeros((self.lag_order, self.n_vars))
        # this is the true residuals
        self.residuals = self.res[:self.n_vars, :]

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
