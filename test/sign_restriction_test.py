import time

import numpy as np
import pandas as pd

from SignRestriction import SignRestriction

# we simulate the data (using the directional information) and data does not make sense, just for the exercises.
data = pd.read_csv('/Users/fangli/PySVAR/PySVAR/data/data1.csv')
data = np.array(data)[:, 1:]
names = ['var1', 'var2', 'var3', 'var4']
shocks = ['sh1', 'sh3']

# put in the target signs
signs = np.zeros((len(names), len(names)))
signs[0, :] = np.array([1, 0, 1, 0])
signs[1, :] = np.array([1, 0, -1, 0])

# a sign restriction instance
fsr = SignRestriction(var_names=names, shock_names=shocks, target_signs=signs, data=data, data_frequency='Q')

# identify
accept = fsr.identify(h=20, n_rotation=1000, irf_sig=[68, 80, 95], vd_sig=[68, 80, 95], seed=3906, verbose=True)

# plot the IRF and VD
fsr.plot_irf(var_list=['var1', 'var2', 'var3'], shock_list=['sh1', 'sh3'], sigs=[68, 95], with_ci=True)
# fsr.plot_vd(var_list=['var1', 'var2', 'var3'], shock_list=['sh1', 'sh3'])
