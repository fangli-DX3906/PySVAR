import datetime
import random
from typing import Union, Literal, List, Tuple
import numpy as np

from SVAR import SetIdentifiedSVAR


class SignRestriction(SetIdentifiedSVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 target_signs: np.ndarray,
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 date_start: datetime.datetime,
                 date_end: datetime.datetime,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         date_frequency=date_frequency,
                         date_start=date_start,
                         date_end=date_end,
                         constant=constant,
                         info_criterion=info_criterion)
        self.identification = 'sign restriction'
        self.target_signs = target_signs
        self.n_ones = np.sum(self.target_signs == 1)
        self.n_nones = np.sum(self.target_signs == -1)
        self.num_unrestricted = self.n_vars ** 2 - self.n_ones - self.n_nones
        if self.n_ones > self.n_nones:
            self.direction = 'descend'
        else:
            self.direction = 'ascend'

    def sort_row(self, mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = []
        for i in list(range(self.n_vars))[::-1]:
            c.append(2 ** i)

        c = np.array(c)
        mask = c * np.ones((self.n_vars, 1))
        C = np.sum(mask * mat, axis=1)
        idx = np.argsort(C)
        if self.direction == 'descend':
            idx = idx[::-1]

        return idx, mat[idx, :]

    def draw_rotation(self) -> np.ndarray:
        raw_mat = np.random.randn(self.n_vars, self.n_vars)
        Q, R = np.linalg.qr(raw_mat)
        Q = np.sign(np.diag(R)).reshape((-1, 1)) * Q

        return Q

    def identify(self,
                 h: int,
                 n_rotation: int,
                 irf_sig: Union[List[int], int],
                 vd_sig: Union[List[int], int, None],
                 length_to_check: int = 1,
                 seed: Union[bool, int] = False,
                 verbose: bool = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        counter = 0
        total = 0
        self.irf_mat = np.zeros((n_rotation, self.n_vars * self.n_shocks, h + 1))
        self.vd_mat = np.zeros((n_rotation, self.n_vars * self.n_shocks, h + 1))
        self.irfs_mat = np.zeros((n_rotation, self.n_vars * self.n_shocks, self.H))

        while counter < n_rotation:
            total += 1
            D = self.draw_rotation()
            irfs = self._ReducedModel__get_irf(h=h, comp_mat=self.comp_mat, cov_mat=self.cov_mat, rotation=D)
            irf_sign = np.sign(np.sum(irfs[:, :length_to_check], axis=1).reshape((self.n_vars, self.n_vars)))

            idx, sorted_signs = self.sort_row(irf_sign)
            diff_sign = self.target_signs - sorted_signs

            if np.sum(diff_sign ** 2) == self.num_unrestricted:
                counter += 1
                if verbose:
                    print(f'{counter} accepted rotation/{n_rotation} required rotations')

                D = D[:, idx]
                irf_temp = self._ReducedModel__get_irf(h=self.H, comp_mat=self.comp_mat, cov_mat=self.cov_mat,
                                                       rotation=D)
                irf_temp_h = irf_temp[:, :h + 1]
                irf_temp_h_var = irf_temp[:(self.n_vars ** 2 - self.n_diff * self.n_vars), :h + 1]
                self.irfs_mat[counter - 1, :, :] = irf_temp
                self.irf_mat[counter - 1, :, :] = irf_temp_h_var
                vds = self._ReducedModel__get_vd(irfs=irf_temp_h)
                self.vd_mat[counter - 1, :, :] = vds[:(self.n_vars ** 2 - self.n_diff * self.n_vars), :]

        self.irf_cv(irf_sig=irf_sig)
        if vd_sig is None:
            vd_sig = irf_sig
        self.vd_cv(vd_sig=vd_sig)
        self.irfs = np.percentile(self.irfs_mat, 50, axis=0)

        # TODO: incorporate HD

        print('*' * 30)
        print(f'acceptance rate is {counter / total}')
