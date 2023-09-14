import scipy.io as spio
import numpy as np
from ExclusionRestriction import ExclusionRestriction
from RecursiveIdentification import RecursiveIdentification

# replilcate Kilian 2009
oil = spio.loadmat('data/oil.mat')
o = oil['data']
n = ['OilProd', 'REA', 'OilPrice']
s = ['Supply', 'Agg Demand', 'Specific Demand']
e = {(0, 1), (0, 2), (1, 2)}
h = 15

exln = ExclusionRestriction(data=o, var_names=n, shock_names=s, exclusion=e, date_frequency='M', lag_order=24)
recr = RecursiveIdentification(data=o, var_names=n, shock_names=s, date_frequency='M', lag_order=24)

exln.identify()
exln.bootstrap(seed=3906)

recr.identify()
recr.bootstrap(seed=3906)

mdls = [exln, recr]
for m in mdls:
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
    m.plot_irf(h=15, var_list=n, sigs=95, with_ci=True)
