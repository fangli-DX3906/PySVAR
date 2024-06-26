import numpy as np
from typing import Literal


def check_stationary(comp_mat: np.ndarray) -> bool:
    eigen_vals, _ = np.linalg.eig(comp_mat)
    eigen_val_mod = abs(eigen_vals)
    flag = np.any(eigen_val_mod >= 1)
    return not flag


def adjust_stationary(comp_mat: np.ndarray, cov_mat: np.ndarray,
                      n_obs: int, lag_order: int, n_vars: int) -> np.ndarray:
    # algorithm according to Kilian, source: Pope (1990), JTSA

    T = n_obs - lag_order
    nums_sq = n_vars * lag_order
    nums = nums_sq ** 2

    # if cov_mat.shape[0] != lag_order * n_vars:
    #     cov_mat_stuffed = cov_mat
    #     cov_mat = np.zeros((nums_sq, nums_sq))
    #     cov_mat[:n_vars, :n_vars] = cov_mat_stuffed

    mat1 = np.kron(comp_mat, comp_mat)
    mat2 = cov_mat.reshape((nums, -1), order='F')
    vecSIGMAY = np.dot(np.linalg.inv((np.eye(nums) - mat1)), mat2)
    SIGMAY = vecSIGMAY.reshape((nums_sq, nums_sq), order='F')
    I = np.eye(nums_sq)
    B = comp_mat.T

    peigen, _ = np.linalg.eig(comp_mat)
    sumeig = np.zeros((nums_sq, nums_sq)).astype('complex128')

    for eig in peigen:
        sumeig += eig * np.linalg.inv(I - eig * B)

    mat3 = np.linalg.inv(I - np.dot(B, B))
    mat4 = np.linalg.inv(I - B) + np.dot(B, mat3) + sumeig
    bias = np.dot(np.dot(cov_mat, mat4), np.linalg.inv(SIGMAY))
    Abias = -bias / T
    Abias = Abias.real

    bcstab = 9
    delta = 1

    while bcstab >= 1:
        bcA = comp_mat - delta * Abias
        eignTemp, _ = np.linalg.eig(bcA)
        bcmod = abs(eignTemp)

        if np.any(bcmod >= 1):
            bcstab = 1
        else:
            bcstab = 0

        delta -= 0.01

        if delta <= 0:
            bcstab = 0

    return bcA


def estim_sys(data: np.ndarray, lag: int,
              constant: bool = True, adjust: bool = False):
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

    if adjust and check_stationary(comp_mat=comp_mat):
        comp_mat = adjust_stationary(comp_mat=comp_mat, cov_mat=cov_mat,
                                     lag_order=lag, n_obs=t, n_vars=q)

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
        # adjust for stationarity does not affect cov_mat
        _, cov_mat_, _, _, _ = estim_sys(data, lag, constant)
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
