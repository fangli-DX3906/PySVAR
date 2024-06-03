import random
import numpy as np
from typing import Union, Literal, List, Optional

from base_model import Model
from auxillary.bricks import estim_sys


class VAR(Model):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: Optional[List[str]] = None,
                 constant: bool = True,
                 lag_order: Optional[int] = None,
                 max_lag_order: Optional[int] = 8,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 date_frequency: Literal['M', 'Q', 'A'] = 'Q',
                 date_start: str = None):

        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         constant=constant,
                         lag_order=lag_order,
                         max_lag_order=max_lag_order,
                         info_criterion=info_criterion,
                         date_frequency=date_frequency,
                         date_start=date_start)

    def estimate(self):
        self.fit()
        self.irf_point_estimate = self.tools.reduced_var_irf_point_estimate
        self.vd_point_estimate = self.tools.estimate_vd(self.tools.reduced_var_irf_point_estimate)
        self.hd_point_estimate = self.tools.estimate_hd(self.residuals, self.tools.reduced_var_irf_point_estimate)

    def bootstrap(self, n_path: int = 100, seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.irf_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H + 1))
        self.vd_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H + 1))
        self.hd_mat_full = np.zeros((n_path, self.n_vars ** 2, self.H))

        for r in range(n_path):
            y_r = self.make_bootstrap_sample()
            comp_mat_r, cov_mat_r, resid_r, _, _ = estim_sys(data=y_r,
                                                             lag=self.lag_order,
                                                             constant=self.constant)
            cov_mat_r = cov_mat_r[:self.n_vars, :self.n_vars]
            residual_r = resid_r[:self.n_vars, :]
            self.tools.update(data=y_r, comp=comp_mat_r, cov=cov_mat_r)
            irf_r = self.tools.estimate_irf()
            self.irf_mat_full[r, :, :] = irf_r
            self.vd_mat_full[r, :, :] = self.tools.estimate_vd(irfs=irf_r)
            self.hd_mat_full[r, :, :] = self.tools.estimate_hd(shocks=residual_r, irfs=irf_r)
