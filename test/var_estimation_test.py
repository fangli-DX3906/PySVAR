import scipy.io as spio
import numpy as np
import pandas as pd
import datetime as dt
from estimation.VAR import VAR

# EX 1: estimate the VAR model
data = spio.loadmat('/Users/fangli/PySVAR/PySVAR/data/estimation_test.mat')
data = data['Y']
data = data[:100, [0, 6]]
names = ['Output', 'Inflation']
var = VAR(data=data, var_names=names, date_frequency='Q',
          date_start=dt.datetime(1990, 1, 1), date_end=dt.datetime(2014, 12, 31))
var.lag_order = 8
var.fit()

# estimate the IRF and VD
h = 20
irf = var.irf(h=h)
vd = var.vd(h=h)
var.bootstrap(h=h, n_path=500)
var.plot_irf(sigs=[68, 80], with_ci=True)
var.plot_vd()

# EX2: replicate Kilian (2009)
oil = spio.loadmat('/Users/fangli/PySVAR/PySVAR/data/oil.mat')
oil = oil['data']
names = ['OilProd', 'REA', 'OilPrice']
shocks = ['Supply', 'Agg Demand', 'Specific Demand']
exm = VAR(data=oil, var_names=names, date_frequency='M',
          date_start=dt.datetime(1973, 1, 1), date_end=dt.datetime(2007, 11, 30))
exm.lag_order = 24
exm.fit()

# estimate the IRF
h = 15
exm.irf(h=h)
exm.vd(h=h)
exm.bootstrap(h=h)
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

# EX3. oil price uncertainty shock
data = pd.read_csv('/Users/fangli/PySVAR/PySVAR/data/data_comp.csv')
data.drop(columns='Unnamed: 0', inplace=True)
var_names = ['VIX_avg', 'OVX_avg', 'WorldOilProd', 'IndustrialProd', 'WTISpotPrice', 'OilInventory']
var = VAR(data=np.array(data[var_names]), var_names=var_names, date_frequency='Q',
          date_start=dt.datetime(1973, 1, 1), date_end=dt.datetime(2007, 11, 30))
var.lag_order = 8
var.fit()

var.irf(h=20)
var.vd(h=20)
var.bootstrap(h=20)
var.plot_irf(var_list=var_names, shock_list=[1], sigs=[68, 80], with_ci=True)
var.plot_vd(var_list=var.var_names, shock_list=[0, 1])
