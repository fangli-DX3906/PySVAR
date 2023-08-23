import scipy.io as spio
import numpy as np
import datetime as dt
from estimation.VAR import VAR

# replicate Kilian (2009)
oil = spio.loadmat('./data/oil.mat')
oil = oil['data']
names = ['OilProd', 'REA', 'OilPrice']
shocks = ['Supply', 'Agg Demand', 'Specific Demand']
exm = VAR(data=oil, var_names=names, date_frequency='M', lag_order=24,
          date_start=dt.datetime(1973, 1, 1), date_end=dt.datetime(2007, 11, 30))

# estimate the IRF
h = 15
exm.irf(h=h)
exm.vd(h=h)
exm.bootstrap(h=h, seed=3906)
exm.irf_point_estimate[0, :] = -np.cumsum(exm.irf_point_estimate[0, :])
exm.irf_point_estimate[3, :] = np.cumsum(exm.irf_point_estimate[3, :])
exm.irf_point_estimate[6, :] = np.cumsum(exm.irf_point_estimate[6, :])
exm.irf_point_estimate[1, :] = -exm.irf_point_estimate[1, :]
exm.irf_point_estimate[2, :] = -exm.irf_point_estimate[2, :]

for _ in range(exm.irf_mat.shape[0]):
    exm.irf_mat[_, 0, :] = -np.cumsum(exm.irf_mat[_, 0, :])
    exm.irf_mat[_, 3, :] = np.cumsum(exm.irf_mat[_, 3, :])
    exm.irf_mat[_, 6, :] = np.cumsum(exm.irf_mat[_, 6, :])
    exm.irf_mat[_, 1, :] = -exm.irf_mat[_, 1, :]
    exm.irf_mat[_, 2, :] = -exm.irf_mat[_, 2, :]

exm.plot_irf(var_list=names, sigs=[68, 95], with_ci=True)
