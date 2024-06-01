from typing import Literal, Optional
import numpy as np


class Estimation:
    def __init__(self,
                 data: np.ndarray,
                 constant: bool):
        self.data = data
        self.constant = constant

    def estimate(self,
                 y: Optional[np.ndarray],
                 lag: int):
        if y is None:
            y = self.data
        t, q = y.shape
        y = y.T
        yy = y[:, lag - 1:t]
        # algorithm according to Kilian
        for i in range(1, lag):
            yy = np.concatenate((yy, y[:, lag - i - 1:t - i]), axis=0)

        if self.constant:
            x = np.concatenate((np.ones((1, t - lag)), yy[:, :t - lag]), axis=0)
        else:
            x = yy[:, :t - lag]

        y = yy[:, 1:t - lag + 1]
        comp_mat = np.dot(np.dot(y, x.T), np.linalg.inv((np.dot(x, x.T))))
        resid = y - np.dot(comp_mat, x)
        cov_mat = np.dot(resid, resid.T) / (t - lag - lag * q - 1)
        constant = comp_mat[:, 0]
        comp_mat = comp_mat[:, 1:]  # comp_mat does not include the intercept
        return comp_mat, cov_mat, resid, constant, x

    def optim_lag(self,
                  y: Optional[np.ndarray],
                  criterion: Literal['aic', 'bic', 'hqc'],
                  max_lags: int = 8):
        if y is None:
            y = self.data
        t, q = y.shape
        aic = []
        bic = []
        hqc = []
        for lag in range(1, max_lags + 1):
            phim = q ** 2 * lag + q
            _, cov_mat_, _, _, _ = self.estimate(y, lag)
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
