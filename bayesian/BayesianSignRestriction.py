from typing import Union, Literal, List
import numpy as np
import datetime

from identification.SignRestriction import SignRestriction
from bayesian.Bayesian import DiffusePrior, MinnesotaPrior


class BayesianSignRestriction(SignRestriction):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 target_signs: np.ndarray,
                 prior: Literal['diffuse', 'Minnesota'],
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 data_frequency: Literal['D', 'W', 'M', 'Q', 'SA', 'A'] = 'Q',
                 date_range: List[datetime.date] = None):
        super().__init__(data, var_names, shock_names, target_signs, constant, info_criterion, data_frequency,
                         date_range)
        if prior == 'diffuse':
            self.posteriors = DiffusePrior(self.likelihood_info)
        else:
            minnesota_dict = {'lambda1': 1, 'lambda2': 1, 'lambda3': 1, 'lambda4': 1, 'lag': 1}
            self.posteriors = MinnesotaPrior(self.data, self.likelihood_info, minnesota_dict)

    def identify(self,
                 h: int,
                 n_burn: int,
                 n_sims: int,
                 n_rotation: int,
                 irf_sig: Union[List[int], int],
                 vd_sig: Union[List[int], int, None],
                 length_to_check: int = 1,
                 seed: Union[bool, int] = False):
        if seed:
            np.random.seed(seed)

        counter = 0
        n_total = n_burn + n_sims
        self.irf_mat = np.zeros((n_sims * n_rotation, self.n_vars * self.n_shocks, h + 1))
        self.vd_mat = np.zeros((n_sims * n_rotation, self.n_vars * self.n_shocks, h + 1))

        cov = self.cov_mat
        for _ in range(n_total):
            mn_1, mn_2 = self.posteriors.get_posterior_comp_dist_params(cov)
            iw_1, iw_2 = self.posteriors.get_posterior_cov_dist_params()
            comp = self.posteriors.draw_comp_from_posterior(mn_1, mn_2)
            cov = self.posteriors.draw_cov_from_posterior(iw_1, iw_2)
            if _ > n_burn:
                counter_for_each_draw = 0
                while counter_for_each_draw < n_rotation:
                    D = self.draw_rotation()
                    irfs = self._ReducedModel__get_irf(h=h, comp_mat=comp, cov_mat=cov, rotation=D)
                    irf_sign = np.sign(np.sum(irfs[:, :length_to_check], axis=1).reshape((self.n_vars, self.n_vars)))
                    idx, sorted_signs = self.sort_row(irf_sign)
                    diff_sign = self.target_signs - sorted_signs
                    if np.sum(diff_sign ** 2) == self.num_unrestricted:
                        counter += 1
                        counter_for_each_draw += 1
                        print(f'{counter} accepted rotation/{n_rotation * n_sims} required rotations')
                        D = D[:, idx]
                        irfs = self._ReducedModel__get_irf(h=h, comp_mat=self.comp_mat, cov_mat=self.cov_mat,
                                                           rotation=D)
                        self.irf_mat[counter - 1, :, :] = irfs[:(self.n_vars ** 2 - self.n_diff * self.n_vars), :]
                        vds = self._ReducedModel__get_vd(irfs=irfs)
                        self.vd_mat[counter - 1, :, :] = vds[:(self.n_vars ** 2 - self.n_diff * self.n_vars), :]

        self.irf_cv(irf_sig=irf_sig, irf_mat=self.irf_mat, median_as_point_estimate=True)
        if vd_sig is None:
            vd_sig = irf_sig
        self.vd_cv(vd_sig=vd_sig, vd_mat=self.vd_mat, median_as_point_estimate=True)
