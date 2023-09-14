import numpy as np
import datetime as dt
import scipy.io as spio
from time import perf_counter

from SignRestriction import SignRestriction

# replicate Marco Brianti (2023)
data = spio.loadmat('data/sign_res.mat')
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
fsr = SignRestriction(data=data,
                      var_names=names,
                      shock_names=shocks,
                      target_signs=signs,
                      date_start=dt.datetime(1982, 1, 1),
                      date_end=dt.datetime(2020, 3, 31),
                      date_frequency='Q')
t0 = perf_counter()
fsr.identify(n_rotation=100, seed=3906, verbose=True)
t1 = perf_counter()
print(f'running time: {t1 - t0} seconds')
_ = fsr.irf(h=40)

fsr.plot_irf(h=40, var_list=var_plot, shock_list=shock_plot, sigs=[68, 95])
