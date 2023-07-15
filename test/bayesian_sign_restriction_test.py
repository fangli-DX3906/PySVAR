import scipy.io as spio
import numpy as np
from time import perf_counter

from bayesian.BayesianSignRestriction import BayesianSignRestriction
from identification.SignRestriction import SignRestriction

data = spio.loadmat('../data/estimation_test2.mat')
data = data['y']
names = ['Output', 'TFP', 'Investment', 'StockReturn', 'Cash', 'CreditSpread', 'Uncertainty', 'Inflation']
shocks = ['Supply', 'Financial', 'Uncertainty', 'Residual']
data[:, 2] = data[:, 2] - data[:, 0]
data[:, 1] = data[:, 1] - data[:, 0]
signs = np.zeros((len(names), len(names)))
signs[0, :] = np.array([-1, -1, 0, 0, 0, 0, 0, 1])  # a negative supply shock
signs[1, :] = np.array([-1, 1, -1, -1, -1, 1, 1, 1])  # financial shock
signs[2, :] = np.array([-1, 1, -1, -1, 1, 1, 1, 0])  # uncertainty shock
signs[3, :] = np.array([0, 0, 1, 0, 0, 0, 0, -1])  # residual shock
var_plot = ['Cash', 'CreditSpread', 'Uncertainty', 'Inflation', 'Output', 'Investment']
shock_plot = ['Financial', 'Uncertainty']

# sign restriction
# fsr = SignRestriction(data=data, var_names=names, shock_names=shocks, target_signs=signs)
# t0 = perf_counter()
# fsr.identify(h=20, n_rotation=100, irf_sig=95, vd_sig=None, seed=3906, verbose=True)
# t1 = perf_counter()
# fsr.plot_irf(var_list=var_plot, shock_list=shock_plot, sigs=95, with_ci=True)

# bayesian sign restriction
bsr = BayesianSignRestriction(data=data, var_names=names, shock_names=shocks, target_signs=signs, prior='diffuse')
t2 = perf_counter()
bsr.identify(h=20, n_burn=1000, n_sims=50, n_rotation=2, irf_sig=95, vd_sig=None, seed=3907)
t3 = perf_counter()
bsr.plot_irf(var_list=var_plot, shock_list=shock_plot, sigs=95, with_ci=True)

# TODO:
# bsr = BayesianSignRestriction(data=data, var_names=names, shock_names=shocks, target_signs=signs, prior='Minnesota')
# t4 = perf_counter()
# bsr.identify(h=20, n_burn=1000, n_sims=50, n_rotation=2, irf_sig=95, vd_sig=None, seed=3907)
# t5 = perf_counter()
# bsr.plot_irf(var_list=var_plot, shock_list=shock_plot, sigs=95, with_ci=True)

# print(f'sign restriction sampling 100 rotations takes {t1 - t0} seconds')
# print(f'bayesian sign restriction sampling 100 rotations takes {t3 - t2} seconds')
