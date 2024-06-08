import scipy.io as spio
import numpy as np
from time import perf_counter

from bayesian.bayesian_var import BayesianVAR

oil = spio.loadmat('./data/oil.mat')
oil = oil['data']
names = ['OilProd', 'REA', 'OilPrice']
shocks = ['Supply', 'Agg Demand', 'Specific Demand']
h = 15
p = 24


def agg_irf(model):
    model.irf_point_estimate[0, :] = -np.cumsum(model.irf_point_estimate[0, :])
    model.irf_point_estimate[3, :] = np.cumsum(model.irf_point_estimate[3, :])
    model.irf_point_estimate[6, :] = np.cumsum(model.irf_point_estimate[6, :])
    model.irf_point_estimate[1, :] = -model.irf_point_estimate[1, :]
    model.irf_point_estimate[2, :] = -model.irf_point_estimate[2, :]
    for _ in range(model.irf_mat_full.shape[0]):
        model.irf_mat_full[_, 0, :] = -np.cumsum(model.irf_mat_full[_, 0, :])
        model.irf_mat_full[_, 3, :] = np.cumsum(model.irf_mat_full[_, 3, :])
        model.irf_mat_full[_, 6, :] = np.cumsum(model.irf_mat_full[_, 6, :])
        model.irf_mat_full[_, 1, :] = -model.irf_mat_full[_, 1, :]
        model.irf_mat_full[_, 2, :] = -model.irf_mat_full[_, 2, :]


# diffuse prior
dp = BayesianVAR(data=oil, var_names=names, date_frequency='M', lag_order=p, prior='Diffuse')
dp.estimate()
t1 = perf_counter()
dp.bayesian_bootstrap(n_burn=1000, n_sims=100, seed=3906)
t2 = perf_counter()
print(f'Running time {t2 - t1}')
agg_irf(dp)
dp.plot_irf(h=h, var_list=names, sigs=[68, 95])

# Minnesota prior-1
prior_params = {'lambda1': 0.2, 'lambda2': 1, 'lambda3': 1, 'lambda4': 10 ** 5, 'comp_mode': 'RandomWalk'}
mp1 = BayesianVAR(data=oil, var_names=names, shock_names=shocks, lag_order=p,
                  prior='Minnesota', prior_params=prior_params, date_frequency='Q')
mp1.estimate()
t3 = perf_counter()
mp1.bayesian_bootstrap(n_burn=1000, n_sims=100, seed=3906)
t4 = perf_counter()
print(f'Running time {t4 - t3}')
agg_irf(mp1)
mp1.plot_irf(h=h, var_list=names, sigs=[68, 95])

# Minnesota prior-2
prior_params = {'lambda1': 0.2, 'lambda2': 1, 'lambda3': 1, 'lambda4': 10 ** 5, 'comp_mode': 'AR1'}
mp2 = BayesianVAR(data=oil, var_names=names, shock_names=shocks, lag_order=p,
                  prior='Minnesota', prior_params=prior_params, date_frequency='Q')
mp2.estimate()
t5 = perf_counter()
mp2.bayesian_bootstrap(n_burn=1000, n_sims=100, seed=3906)
t6 = perf_counter()
print(f'Running time {t6 - t5}')
agg_irf(mp2)
mp2.plot_irf(h=h, var_list=names, sigs=[68, 95])

# natural conjugate prior
prior_params = {'lambda1': 0.2, 'lambda2': 1, 'lambda3': 1, 'lambda4': 10 ** 4, 'comp_mode': 'AR1'}
ncp = BayesianVAR(data=oil, var_names=names, shock_names=shocks, lag_order=p,
                  prior='NaturalConjugate', prior_params=prior_params, date_frequency='Q')
ncp.estimate()
t7 = perf_counter()
ncp.bayesian_bootstrap(n_burn=1000, n_sims=100, seed=3906)
t8 = perf_counter()
print(f'Running time {t8 - t7}')
agg_irf(ncp)
ncp.plot_irf(h=h, var_list=names, sigs=[68, 95])

# nomral diffuse prior
prior_params = {'lambda1': 0.2, 'lambda2': 1, 'lambda3': 1, 'lambda4': 10 ** 5, 'comp_mode': 'AR1'}
ndp = BayesianVAR(data=oil, var_names=names, shock_names=shocks, lag_order=p,
                  prior='NormalDiffuse', prior_params=prior_params, date_frequency='Q')
ndp.estimate()
t9 = perf_counter()
ndp.bayesian_bootstrap(n_burn=1000, n_sims=100, seed=3906)
t10 = perf_counter()
print(f'Running time {t10 - t9}')
agg_irf(ndp)
ndp.plot_irf(h=h, var_list=names, sigs=[68, 95])
