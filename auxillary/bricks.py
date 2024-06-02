from typing import Literal
import numpy as np


def estimate_var(data: np.ndarray, lag: int, constant: bool = True):
    t, q = data.shape
    y = data.T
    yy = y[:, lag - 1:t]

    # algorithm according to Kilian
    for i in range(1, lag):
        yy = np.concatenate((yy, y[:, lag - i - 1:t - i]), axis=0)

    if constant:
        x = np.concatenate((np.ones((1, t - lag)), yy[:, :t - lag]), axis=0)
    else:
        x = yy[:, :t - lag]

    y = yy[:, 1:t - lag + 1]
    comp_mat = np.dot(np.dot(y, x.T), np.linalg.inv((np.dot(x, x.T))))
    resid = y - np.dot(comp_mat, x)
    cov_mat = np.dot(resid, resid.T) / (t - lag - lag * q - 1)

    if constant:
        # comp_mat does not include the intercept
        constant = comp_mat[:, 0]
        comp_mat = comp_mat[:, 1:]
    else:
        constant = None

    return comp_mat, cov_mat, resid, constant, x


def optim_lag(data: np.ndarray,
              criterion: Literal['aic', 'bic', 'hqc'],
              max_lags: int = 8,
              constant: bool = True):
    t, q = data.shape
    aic = []
    bic = []
    hqc = []
    for lag in range(1, max_lags + 1):
        phim = q ** 2 * lag + q
        _, cov_mat_, _, _, _ = estimate_var(data, lag, constant)
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
