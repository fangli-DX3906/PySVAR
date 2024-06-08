import random
import numpy as np
from tqdm import tqdm
from typing import List, Literal, Optional, Union

from core.var import VAR
from bayesian.posterior_generator import PosteriorGenerator


class BayesianVAR(VAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: Optional[List[str]] = None,
                 constant: bool = True,
                 prior: Literal['Diffuse', 'NormalDiffuse', 'Minnesota', 'NaturalConjugate'] = 'Diffuse',
                 prior_params: Optional[dict] = None,
                 lag_order: Optional[int] = None,
                 max_lag_order: Optional[int] = 8,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 date_frequency:
                 Literal['M', 'Q', 'A'] = 'Q',
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
        self.fit()
        self.posterior_generator = PosteriorGenerator(likelihood_info=self.likelihood_info,
                                                      prior_name=prior + 'Prior',
                                                      prior_params=prior_params)

    def estimate(self):
        self.irf_point_estimate = self.tools.reduced_var_irf_point_estimate
        self.vd_point_estimate = self.tools.estimate_vd(self.tools.reduced_var_irf_point_estimate)
        self.hd_point_estimate = self.tools.estimate_hd(self.residuals, self.tools.reduced_var_irf_point_estimate)

    def bayesian_bootstrap(self,
                           n_burn: int,
                           n_sims: int,
                           how: Literal['median', 'average'] = 'median',
                           # parallel: bool = False,
                           # n_process: int = 4,
                           seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        n_total = n_burn + n_sims
        pbar = tqdm(total=n_sims, desc=f'Simulating {n_sims} samples...')
        cov = self.cov_mat
        y = self.likelihood_info['y']
        X = self.likelihood_info['X']

        self.irf_mat_full = np.zeros((n_sims, self.n_vars ** 2, self.H + 1))
        self.vd_mat_full = np.zeros((n_sims, self.n_vars ** 2, self.H + 1))
        self.hd_mat_full = np.zeros((n_sims, self.n_vars ** 2, self.H))

        for _ in range(n_total):
            comp, cov = self.posterior_generator.draw_from_posterior(sigma=cov)

            if _ >= n_burn:
                r = _ - n_burn
                pbar.update(1)
                comp = self.posterior_generator.recover_comp_mat(comp)
                self.tools.update(comp=comp, cov=cov)
                irf_r = self.tools.estimate_irf()
                self.irf_mat_full[r, :, :] = irf_r
                self.vd_mat_full[r, :, :] = self.tools.estimate_vd(irfs=irf_r)
                # residual_r = y - np.dot(X.T, comp)
                residual_r = self.residuals
                self.hd_mat_full[r, :, :] = self.tools.estimate_hd(shocks=residual_r, irfs=irf_r)
