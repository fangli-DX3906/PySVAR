import scipy.io as spio
import numpy as np

from estimation.var import VAR

# replicate Kilian (2009)
oil = spio.loadmat('/Users/fangli/PySVAR/PySVAR/data/oil.mat')
oil = oil['data']
names = ['OilProd', 'REA', 'OilPrice']
shocks = ['Supply', 'Agg Demand', 'Specific Demand']
m = VAR(data=oil, var_names=names, date_frequency='M', lag_order=4, date_start='197301')
m.estimate()

# estimate the IRF (Figure 3)
h = 15
m.bootstrap(seed=3906)
m.irf_point_estimate[0, :] = -np.cumsum(m.irf_point_estimate[0, :])
m.irf_point_estimate[3, :] = np.cumsum(m.irf_point_estimate[3, :])
m.irf_point_estimate[6, :] = np.cumsum(m.irf_point_estimate[6, :])
m.irf_point_estimate[1, :] = -m.irf_point_estimate[1, :]
m.irf_point_estimate[2, :] = -m.irf_point_estimate[2, :]

for _ in range(m.irf_mat_full.shape[0]):
    m.irf_mat_full[_, 0, :] = -np.cumsum(m.irf_mat_full[_, 0, :])
    m.irf_mat_full[_, 3, :] = np.cumsum(m.irf_mat_full[_, 3, :])
    m.irf_mat_full[_, 6, :] = np.cumsum(m.irf_mat_full[_, 6, :])
    m.irf_mat_full[_, 1, :] = -m.irf_mat_full[_, 1, :]
    m.irf_mat_full[_, 2, :] = -m.irf_mat_full[_, 2, :]

m.plot_irf(h=h, var_list=names, sigs=[68, 95], save_path='./graphs')
m.plot_vd(h=h, var_list=names)
