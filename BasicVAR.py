from typing import Literal, List, Tuple
import numpy as np
import datetime


class BasicVARModel:
    def __init__(self,
                 y: np.ndarray,
                 var_names: list,
                 data_frequency: Literal['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annually', 'Annually'],
                 date_range: List[datetime.date] = None,  # specific to HD
                 constant: bool = True):
        self.data = y
        self.n_obs, self.n_vars = self.data.shape
        self.var_names = var_names
        if self.n_vars != len(self.var_names):
            raise ValueError('Names are not consistent with data dimension!')
        self.fit_constant = constant
        self.date_range = date_range
        self.data_frequency = data_frequency

    def optim_lag_order(self,
                        y: np.ndarray,
                        criterion: Literal['aic', 'bic', 'hqc'],
                        max_lags: int = 8) -> int:
        t, q = y.shape
        aic = []
        bic = []
        hqc = []
        for lag in range(1, max_lags + 1):
            phim = q ** 2 * lag + q
            _, cov_mat_, _, _, _ = self._fit(y, lag)
            sigma = cov_mat_[:q, :q]
            aic.append(np.log(np.linalg.det(sigma)) + 2 * phim / t)
            bic.append(np.log(np.linalg.det(sigma)) + phim * np.log(t) / t)
            hqc.append(np.log(np.linalg.det(sigma)) + 2 * phim * np.log(np.log(t)) / t)
        if criterion == 'aic':
            return np.argmin(aic) + 1
        elif criterion == 'bic':
            return np.argmin(bic) + 1
        else:
            return np.argmin(hqc) + 1

    def _fit(self,
             y: np.ndarray,
             lag: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t, q = y.shape
        y = y.T
        yy = y[:, lag - 1:t]
        # stacking the data according to Kilian p30
        for i in range(1, lag):
            yy = np.concatenate((yy, y[:, lag - i - 1:t - i]), axis=0)
        if self.fit_constant:
            x = np.concatenate((np.ones((1, t - lag)), yy[:, :t - lag]), axis=0)
        else:
            x = yy[:, :t - lag]

        y = yy[:, 1:t - lag + 1]
        comp_mat = np.dot(np.dot(y, x.T), np.linalg.inv((np.dot(x, x.T))))
        resid = y - np.dot(comp_mat, x)
        cov_mat = np.dot(resid, resid.T) / (t - lag - lag * q - 1)
        constant = comp_mat[:, 0]
        comp_mat = comp_mat[:, 1:]

        return comp_mat, cov_mat, resid, constant, x

    def fit(self,
            criterion: Literal['aic', 'bic', 'hqc'] = 'aic') -> None:
        self.lag_order = self.optim_lag_order(self.data, criterion)
        self.comp_mat, cov_mat_, self.resids, self._intercepts, self._x = self._fit(self.data, self.lag_order)
        self.cov_mat = cov_mat_[:self.n_vars, :self.n_vars]
        self.intercepts = self._intercepts[:self.n_vars]
        self.ar_coeff = dict()

        for i in range(0, self.lag_order):
            self.ar_coeff[str(i + 1)] = self.comp_mat[:self.n_vars, i * self.n_vars:(i + 1) * self.n_vars]
