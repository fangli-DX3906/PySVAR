import scipy.io as spio
import numpy as np
from time import perf_counter

from bayesian.bayesian_sign_restriction import BayesianSignRestriction
from identification.sign_restriction import SignRestriction

data = spio.loadmat('./data/sign_res.mat')
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

# frequentest
fsr = SignRestriction(data=data,
                      var_names=names,
                      shock_names=shocks,
                      target_signs=signs,
                      date_frequency='Q')
t0 = perf_counter()
fsr.identify(n_rotation=200, parallel=True, seed=3906)
t1 = perf_counter()
fsr.plot_irf(h=40, var_list=var_plot, shock_list=shock_plot, sigs=[68, 95])

# bayesian
bsr = BayesianSignRestriction(data=data,
                              var_names=names,
                              shock_names=shocks,
                              target_signs=signs,
                              prior='Diffuse',
                              date_frequency='Q')
t2 = perf_counter()
bsr.identify(n_burn=1000, n_sims=100, n_rotation=2, seed=3906)
t3 = perf_counter()
bsr.plot_irf(h=40, var_list=var_plot, shock_list=shock_plot, sigs=[68, 95])
