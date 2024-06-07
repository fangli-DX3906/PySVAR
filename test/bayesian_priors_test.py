import scipy.io as spio
import numpy as np
from time import perf_counter

from bayesian.bayesian_sign_restriction import BayesianSignRestriction

data = spio.loadmat('./data/sign_res.mat')
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

# diffuse prior
dp = BayesianSignRestriction(data=data, var_names=names, shock_names=shocks,
                             target_signs=signs, prior='Diffuse', date_frequency='Q')
t1 = perf_counter()
dp.identify(n_burn=1000, n_sims=100, n_rotation=2, seed=3906)
t2 = perf_counter()
print(f'Running time {t2 - t1}')
dp.plot_irf(h=40, var_list=var_plot, shock_list=shock_plot, sigs=[68, 95])

# Minnesota prior-1
prior_params = {'lambda1': 0.2, 'lambda2': 1, 'lambda3': 1, 'lambda4': 10 ** 5, 'comp_mode': 'RandomWalk'}
mp1 = BayesianSignRestriction(data=data, var_names=names, shock_names=shocks, target_signs=signs,
                              prior='Minnesota', prior_params=prior_params, date_frequency='Q')
t3 = perf_counter()
mp1.identify(n_burn=1000, n_sims=100, n_rotation=2, seed=3906)
t4 = perf_counter()
print(f'Running time {t4 - t3}')
mp1.plot_irf(h=40, var_list=var_plot, shock_list=shock_plot, sigs=[68, 95])

# Minnesota prior-2
prior_params = {'lambda1': 0.2, 'lambda2': 1, 'lambda3': 1, 'lambda4': 10 ** 5, 'comp_mode': 'AR1'}
mp2 = BayesianSignRestriction(data=data, var_names=names, shock_names=shocks, target_signs=signs,
                              prior='Minnesota', prior_params=prior_params, date_frequency='Q')
t5 = perf_counter()
mp2.identify(n_burn=1000, n_sims=100, n_rotation=2, seed=3906)
t6 = perf_counter()
print(f'Running time {t6 - t5}')
mp2.plot_irf(h=40, var_list=var_plot, shock_list=shock_plot, sigs=[68, 95])

# natural conjugate prior
prior_params = {'lambda1': 0.2, 'lambda2': 1, 'lambda3': 1, 'lambda4': 10 ** 5, 'comp_mode': 'AR1'}
ncp = BayesianSignRestriction(data=data, var_names=names, shock_names=shocks, target_signs=signs,
                              prior='NaturalConjugate', prior_params=prior_params, date_frequency='Q')
t7 = perf_counter()
ncp.identify(n_burn=1000, n_sims=100, n_rotation=2, seed=3906)
t8 = perf_counter()
print(f'Running time {t8 - t7}')
ncp.plot_irf(h=40, var_list=var_plot, shock_list=shock_plot, sigs=[68, 95])

# nomral diffuse prior
prior_params = {'lambda1': 0.2, 'lambda2': 1, 'lambda3': 1, 'lambda4': 10 ** 5, 'comp_mode': 'AR1'}
ndp = BayesianSignRestriction(data=data, var_names=names, shock_names=shocks, target_signs=signs,
                              prior='NormalDiffuse', prior_params=prior_params, date_frequency='Q')
t9 = perf_counter()
ndp.identify(n_burn=1000, n_sims=100, n_rotation=2, seed=3906)
t10 = perf_counter()
print(f'Running time {t10 - t9}')
ndp.plot_irf(h=40, var_list=var_plot, shock_list=shock_plot, sigs=[68, 95])
