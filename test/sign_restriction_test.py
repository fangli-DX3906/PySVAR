import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf

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
signs[0, :] = np.array([1, 1, 1, 1, 1, 0])
signs[1, :] = np.array([-1, 1, 1, 1, 1, 0])
signs[2, :] = np.array([0, 1, -1, 1, 1, 0])
signs[3, :] = np.array([0, -1, 0, 1, 1, 0])
signs[4, :] = np.array([1, 0, 0, -1, 1, 0])
fsr = SignRestriction(data=data,
                      var_names=names,
                      shock_names=shocks,
                      target_signs=signs,
                      date_start=dt.datetime(1985, 3, 31),
                      date_end=dt.datetime(2013, 6, 30),
                      date_frequency='Q')
accept = fsr.identify(h=20, n_rotation=200, irf_sig=[68, 95], vd_sig=[68, 95], seed=3906, verbose=True)
fsr.plot_irf(var_list=names, shock_list=['financial'], sigs=[68, 95], with_ci=True)
