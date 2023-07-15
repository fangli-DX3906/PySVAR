import numpy as np


def draw_inverse_whishart_distribution(S, nu):
    S = np.linalg.inv(S)
    A = np.dot(np.linalg.cholesky(S), np.random.randn(S.shape[0], nu))
    return np.linalg.inv(np.dot(A, A.T))


def draw_multi_normal_distribution(psi, omega):
    n_params = psi.shape[0]
    A = psi + np.dot(np.linalg.cholesky(omega), np.random.randn(n_params, 1))
    return A
