import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import scipy.io as spio
from time import perf_counter

from SignRestriction import SignRestriction


# # EX1: we simulate the data (using the directional information) and data does not make sense, just for the exercises.
# data = pd.read_csv('/Users/fangli/PySVAR/PySVAR/data/data1.csv')
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
# accept = fsr.identify(h=20, n_rotation=100, irf_sig=[68, 80, 95], vd_sig=[68, 80, 95], seed=3906, verbose=True)
# fsr.plot_irf(var_list=['A', 'B', 'C', 'D'], shock_list=['demand', 'supply'], sigs=[68, 95], with_ci=True)


# EX2: replicate Furlanetto et al. (2014)
def find_month(x):
    if x == 0:
        return 3
    elif x == 0.25:
        return 6
    elif x == 0.5:
        return 9
    else:
        return 12


# import the data
data = pd.read_excel('/Users/fangli/PySVAR/PySVAR/data/FFS.xlsx')
data = data.loc[:114, :]
data['year'] = data.time.apply(lambda x: int(x))
data['flag'] = data.time - data.year
data['month'] = data.flag.apply(find_month)
data = data.iloc[:data.shape[0] - 1, :]

spn = yf.download('^GSPC')
spn['date'] = spn.index
spn = spn.loc[(spn.date > dt.datetime(1985, 1, 1)) & (spn.date < dt.datetime(2013, 9, 10))]
spn['month'] = spn.date.apply(lambda x: x.month)
spn['year'] = spn.date.apply(lambda x: x.year)
spn['flag'] = np.nan
spn = spn.loc[(spn.month == 3) | (spn.month == 6) | (spn.month == 9) | (spn.month == 12)]
for idx in range(0, spn.shape[0] - 1):
    if spn.month.iloc[idx] == spn.month.iloc[idx + 1]:
        pass
    else:
        spn.flag.iloc[idx] = 1
spn = spn.loc[spn.flag == 1]

data = pd.merge(left=data, right=spn, on=['year', 'month'])
data = data[['RealGDP', 'RealInvestment', 'FFR', 'GDPDeflator', 'Baa', 'Yield', 'Adj Close']]

# data cleaning
data['gdp'] = np.log(data['RealGDP'])
data['deflator'] = np.log(data['GDPDeflator'])
data['int'] = data['Yield']
data['inv'] = np.log(data['RealInvestment'])
data['stock'] = np.log(data['Adj Close'])
data['spread'] = data['Baa'] - data['FFR']
data = data[['gdp', 'deflator', 'int', 'inv', 'stock', 'spread']]
data = data.to_numpy()

# sign restriction
names = ['GDP', 'Deflator', 'InterestRate', 'Investment', 'StockPrice', 'Spread']
shocks = ['supply', 'demand', 'monetary', 'investment', 'financial']
signs = np.zeros((len(names), len(names)))
signs[0, :] = np.array([1, -1, 0, 0, 1, 0])  # supply
signs[1, :] = np.array([-1, 1, 1, 1, 0, 0])  # demand
signs[2, :] = np.array([1, 1, -1, 0, 0, 0])  # monetary
signs[3, :] = np.array([1, 1, 1, 1, -1, 0])  # investment
signs[4, :] = np.array([1, 1, 1, 1, 1, 0])   # financial
fsr = SignRestriction(data=data,
                      var_names=names,
                      shock_names=shocks,
                      target_signs=signs,
                      date_start=dt.datetime(1985, 1, 1),
                      date_end=dt.datetime(2013, 6, 30),
                      date_frequency='Q')
accept = fsr.identify(h=20, n_rotation=100, irf_sig=[68, 95], vd_sig=[68, 95], seed=3906, verbose=True)
fsr.plot_irf(var_list=names, shock_list=['supply', 'demand', 'monetary'], sigs=[68, 95], with_ci=True)

# EX3: replicate Marco Brianti (2023)
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
fsr.identify(h=20, n_rotation=500, irf_sig=[68, 95], vd_sig=None, seed=3906, verbose=True)
t1 = perf_counter()
fsr.plot_irf(var_list=var_plot, shock_list=shock_plot, sigs=[68, 95], with_ci=True)
