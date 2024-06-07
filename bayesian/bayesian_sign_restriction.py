from typing import Union, Literal, Optional
import numpy as np

from identification.sign_restriction import SignRestriction
from bayesian.posterior_generator import PosteriorGenerator


class BayesianSignRestriction(SignRestriction):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 target_signs: np.ndarray,
                 constant: bool = True,
                 prior: Literal['Diffuse', 'NormalDiffuse', 'Minnesota', 'NaturalConjugate'] = 'Diffuse',
                 prior_params: Optional[dict] = None,
                 lag_order: Optional[int] = None,
                 max_lag_order: Optional[int] = 8,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 date_frequency: Literal['M', 'Q', 'A'] = 'Q',
                 date_start: str = None):

        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         target_signs=target_signs,
                         constant=constant,
                         lag_order=lag_order,
                         max_lag_order=max_lag_order,
                         info_criterion=info_criterion,
                         date_frequency=date_frequency,
                         date_start=date_start)

        self.posterior_generator = PosteriorGenerator(likelihood_info=self.likelihood_info,
                                                      prior_name=prior + 'Prior',
                                                      prior_params=prior_params)

    def identify(self,
                 n_burn: int,
                 n_sims: int,
                 n_rotation: int,
                 length_to_check: int = 1,
                 how: Literal['median', 'average'] = 'median',
                 # parallel: bool = False,
                 # n_process: int = 4,
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
                    self.tools.update(rotation=D, comp=comp, cov=cov)
                    _irfs_ = self.tools.estimate_irf(length=length_to_check)
                    irf_sign = np.sign(np.sum(_irfs_, axis=1).reshape((self.n_vars, self.n_vars)))
                    idx, sorted_signs = self._sort_row(irf_sign)
                    diff_sign = self.target_signs - sorted_signs
                    if np.sum(diff_sign ** 2) == self.num_unrestricted:
                        counter += 1
                        counter_for_each_draw += 1
                        D = D[:, idx]
                        rotation_list.append(D)

        self.rotation_list = rotation_list
        self.calc_point_estimate(how=how)
