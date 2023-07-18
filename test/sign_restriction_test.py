import time
import numpy as np
import pandas as pd
import datetime as dt

from SignRestriction import SignRestriction

# EX1: we simulate the data (using the directional information) and data does not make sense, just for the exercises.
data = pd.read_csv('/Users/fangli/PySVAR/PySVAR/data/data1.csv')
data = np.array(data)[:, 1:]
names = ['Shuabd', 'Xydnf', 'Dased', 'Pdnxhfbf']
shocks = ['demand', 'supply']

# put in the target signs
signs = np.zeros((len(names), len(names)))
signs[0, :] = np.array([1, 0, 1, 0])
signs[1, :] = np.array([1, 0, -1, 0])

# a sign restriction instance
fsr = SignRestriction(data=data,
                      var_names=names,
                      shock_names=shocks,
                      target_signs=signs,
                      date_start=dt.datetime(1900, 1, 1),
                      date_end=dt.datetime(1983, 4, 1),
                      date_frequency='M')

# identify
accept = fsr.identify(h=20, n_rotation=100, irf_sig=[68, 80, 95], vd_sig=[68, 80, 95], seed=3906, verbose=True)

# plot the IRF and VD
fsr.plot_irf(var_list=['Shuabd', 'Xydnf', 'Dased', 'Pdnxhfbf'], shock_list=['demand', 'supply'], sigs=[68, 95],
             with_ci=True)

# EX2: replicate Furlanetto et al. (2014)

# EX3: replicate Marco Brianti (2023)
