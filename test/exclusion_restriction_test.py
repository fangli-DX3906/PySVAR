import pandas as pd
import scipy.io as spio
import numpy as np
from VAR import VAR
from ExclusionRestriction import ExclusionRestriction

data = spio.loadmat('/Users/fangli/PySVAR/data/estimation_test.mat')
data = data['Y']
data = data[:100, [0, 6]]
names = ['Output', 'Inflation']

# benchmark
var = VAR(data=data, var_names=names, data_frequency='Q')
var.fit()
irf = var.irf(h=20)
var.bootstrap(h=20, seed=3906)
print(np.linalg.cholesky(var.cov_mat))
var.plot_irf(sigs=[68, 80], with_ci=True)

# test exclusions
shocks = [f'shock{i + 1}' for i in range(2)]
exclusion = {(0, 1)}
exm = ExclusionRestriction(data=data, var_names=names, shock_names=shocks, exclusion=exclusion, data_frequency='Q')
exm.identify(h=20)
exm.boot_confid_intvl(h=20, n_path=100, irf_sig=[68, 80, 95], seed=3906)
print(exm.rotation)
exm.plot_irf(var_list=names, shock_list=['shock1'], sigs=[68, 80], with_ci=True)

# replilcate Kilian 2009
oil = spio.loadmat('/Users/fangli/PySVAR/data/oil.mat')
oil = oil['data']
names = ['OilProd', 'REA', 'OilPrice']
shocks = ['Supply', 'Agg Demand', 'Specific Demand']
exclusion = {(0, 1), (0, 2), (1, 2)}
exm = ExclusionRestriction(data=oil, var_names=names, shock_names=shocks, exclusion=exclusion, data_frequency='Q')

exm.identify(h=20)
print(exm.rotation)
exm.irf_point_estimate = np.cumsum(exm.irf_point_estimate, axis=1)
exm.plot_irf(var_list=names, shock_list=shocks, sigs=[68, 80], with_ci=False)
