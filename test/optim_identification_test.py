import pandas as pd
import numpy as np
from scipy.linalg import null_space

from identification.optim_identification import OptimIdentification

data = pd.read_csv('./data/oil_uncertainty.csv')
var_names = ['OilVol', 'StockVol', 'VolRatio', 'WorldOilProd', 'IndustrialProd', 'WTISpotPrice', 'OilInventory']
data = np.array(data[var_names])
shocks = ['Uncertainty']


# A class derived from the OptimIdentification that implements a target function and two constraints
class OilUncertainty(OptimIdentification):
    def target_function(self, gamma, comp_mat: np.ndarray, cov_mat: np.ndarray):
        gamma = gamma.reshape((1, -1))
        gamma_null = null_space(gamma)
        rotation = np.concatenate((gamma.T, gamma_null), axis=1)
        self.tools.update(rotation=rotation, comp=comp_mat, cov=cov_mat)
        irf = self.tools.estimate_irf(length=3)
        idx = self.var_names.index('OilVol')
        idx_reg = self.var_names.index('VolRatio')
        func = np.sum(irf[idx, :])
        reg_part = 10000 * np.sum(irf[idx_reg, :])

        return -(func + reg_part)

    def constraint_garch_eq(self, gamma):
        return gamma[0]

    def constraint_normalize_eq(self, gamma):
        return np.dot(gamma, gamma) - 1


# make an instance
oil = OilUncertainty(
    data=data,
    var_names=var_names,
    shock_names=shocks,
    constant=True,
    lag_order=6,
    date_frequency='M',
    date_start='200706'
)

oil.identify()
oil.bootstrap(seed=3906)
oil.plot_irf(h=40, var_list=['WorldOilProd', 'IndustrialProd', 'WTISpotPrice', 'OilInventory'],
             shock_list=['Uncertainty'], sigs=[90, 95])
oil.plot_vd(h=40, var_list=['WorldOilProd', 'IndustrialProd', 'WTISpotPrice', 'OilInventory'],
            shock_list=['Uncertainty'])
