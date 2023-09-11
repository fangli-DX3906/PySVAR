import numpy as np
import pandas as pd
import datetime as dt
import scipy.io as spio
from time import perf_counter

from SignRestriction import SignRestriction

# EX1: we simulate the data (using the directional information) and data does not make sense, just for the exercises.
# data = pd.read_csv('./data/data1.csv')
# data = np.array(data)[:, 1:]
# names = ['A', 'B', 'C', 'D']
# shocks = ['demand', 'supply']
#
# # put in the target signs
# signs = np.zeros((len(names), len(names)))
# signs[0, :] = np.array([1, 0, 1, 0])
# signs[1, :] = np.array([1, 0, -1, 0])
#
# # a sign restriction instance
# fsr = SignRestriction(data=data,
#                       var_names=names,
#                       shock_names=shocks,
#                       target_signs=signs,
#                       date_start=dt.datetime(1900, 1, 1),
#                       date_end=dt.datetime(1983, 4, 1),
#                       date_frequency='M')
#
# # identify
# accept = fsr.identify(h=20, n_rotation=100, seed=3906, verbose=True)
# fsr.plot_irf(var_list=['A', 'B', 'C', 'D'], shock_list=['demand', 'supply'], sigs=[68, 95], with_ci=True)

# EX2: replicate Marco Brianti (2023)
data = spio.loadmat('/Users/fangli/PySVAR/PySVAR/data/sign_res.mat')
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
fsr.identify(n_rotation=50, seed=3906, verbose=True)
t1 = perf_counter()
print(f'running: {t1 - t0} seconds')
fsr.irf(h=40)

fsr.plot_irf(h=40, var_list=var_plot, shock_list=shock_plot, sigs=[68, 95])
