from typing import List
import numpy as np
import pandas as pd
import datetime
from statsmodels.tsa.filters.bk_filter import bkfilter as bk
from statsmodels.tsa.filters.hp_filter import hpfilter as hp
from statsmodels.api import OLS
from quantecon import hamilton_filter as ha


class Data:
    def __init__(self,
                 data: np.ndarray,
                 names: List[str],
                 range: List[datetime.date]):
        self.data = pd.DataFrame(data=data, columns=names, index=range)
        self.data['cons'] = 1

    def linear_filter(self,
                      which_var: str):
        self.data['t'] = np.arange(1, self.data.shape[0] + 1)
        lm = OLS(endog=self.data[which_var], exog=self.data[['cons', 't']]).fit()
        self.data[which_var] = lm.resid

    def quadratic_filter(self,
                         which_var: str):
        if not 't' in self.data.columns.tolist():
            self.data['t'] = np.arange(1, self.data.shape[0] + 1)
        self.data['t2'] = self.data['t'] ** 2
        lm = OLS(endog=self.data[which_var], exog=self.data[['cons', 't', 't2']]).fit()
        self.data[which_var] = lm.resid

    def poly_filter(self,
                    which_var: str,
                    order: int):
        if not 't' in self.data.columns.tolist():
            self.data['t'] = np.arange(1, self.data.shape[0] + 1)
        regs = ['cons', 't']
        for i in range(2, order + 1):
            n = f't{i}'
            self.data[n] = self.data['t'] ** i
            regs.append(n)
        lm = OLS(endog=self.data[which_var], exog=self.data[regs]).fit()
        self.data[which_var] = lm.resid

    def hp_filter(self,
                  which_var: str,
                  lam: int = 1600):
        self.data[which_var] = hp(self.data[which_var], lamb=lam)[0]
        self.data.dropna(inplace=True)

    def band_pass_filter(self,
                         which_var: str,
                         low: int = 6,
                         high: int = 32,
                         K: int = 12):
        self.data[which_var] = bk(self.data[which_var], low=low, high=high, K=K)
        self.data.dropna(inplace=True)

    def hamilton_filter(self,
                        which_var: str,
                        h: int = 8,
                        p: int = 4):
        self.data[which_var] = ha(self.data[which_var], h=h, p=p)[0]
        self.data.dropna(inplace=True)
