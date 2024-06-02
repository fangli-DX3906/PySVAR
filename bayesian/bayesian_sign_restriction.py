from typing import Union, Literal, Optional
import numpy as np
import datetime

from identification.sign_restriction import SignRestriction
from bayesian.bayesian import Bayesian


class BayesianSignRestriction(SignRestriction):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 target_signs: np.ndarray,
                 prior: Literal['Diffuse', 'Minnesota'],
                 date_frequency: Literal['M', 'Q', 'A']= None,
                 prior_params: Optional[dict] = None,
                 date_start: Optional[str] = None,
                 lag_order: Optional[int] = None,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        prior_name = prior + 'Prior'
        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         target_signs=target_signs,
                         date_frequency=date_frequency,
                         date_start=date_start,
                         lag_order=lag_order,
                         constant=constant,
                         info_criterion=info_criterion)
        self.posterior_generator = Bayesian(likelihood_info=self.likelihood_info,
                                            prior_name=prior_name,
                                            prior_params=prior_params)

    def identify(self,
                 n_burn: int,
                 n_sims: int,
                 n_rotation: int,
                 # parallel: bool = False,
                 # n_process: int = 4,
                 length_to_check: int = 1,
                 seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)

        rotation_list = []
        counter = 0
        n_total = n_burn + n_sims
        cov = self.cov_mat

        for _ in range(n_total):
            comp, cov = self.posterior_generator.draw_from_posterior(sigma=cov)

            if _ >= n_burn:
                counter_for_each_draw = 0

                while counter_for_each_draw < n_rotation:
                    D = self.draw_rotation()
                    self.tool.update(rotation=D, comp=comp, cov=cov)
                    self.tool.estimate_irf(length=length_to_check)
                    _irfs_ = self.tool.irf
                    irf_sign = np.sign(np.sum(_irfs_, axis=1).reshape((self.n_vars, self.n_vars)))
                    idx, sorted_signs = self._sort_row(irf_sign)
                    diff_sign = self.target_signs - sorted_signs
                    if np.sum(diff_sign ** 2) == self.num_unrestricted:
                        counter += 1
                        counter_for_each_draw += 1
                        D = D[:, idx]
                        rotation_list.append(D)
        self.rotation_list = rotation_list
        self._full_irf()
