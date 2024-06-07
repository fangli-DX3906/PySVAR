import numpy as np
from statsmodels.api import OLS, add_constant


def calc_ar1_coeff(data: np.ndarray):
    ar1_coeff = []
    ar1_sigma = []
    for i in range(data.shape[1]):
        reg = OLS(data[1:, i], add_constant(data[:-1, i])).fit()
        ar1_coeff.append(reg.params[1])
        ar1_sigma.append(reg.mse_resid ** 0.5)

    return ar1_coeff, ar1_sigma
