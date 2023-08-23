import datetime
import random
from typing import Union, Literal, List, Tuple, Optional
import numpy as np

from estimation.SVAR import SVAR
from Tools import Tools


class SignRestriction(SVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 target_signs: np.ndarray,
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 date_start: datetime.datetime,
                 date_end: datetime.datetime,
                 lag_order: Optional[int] = None,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         date_frequency=date_frequency,
                         date_start=date_start,
                         date_end=date_end,
                         set_identified=True,
                         lag_order=lag_order,
                         constant=constant,
                         info_criterion=info_criterion)
        self.tool = Tools(data=data,
                          lag_order=self.lag_order,
                          comp_mat=self.comp_mat,
                          cov_mat=self.cov_mat)
        self.identification = 'sign restriction'
        self.target_signs = target_signs
        self.n_ones = np.sum(self.target_signs == 1)
        self.n_nones = np.sum(self.target_signs == -1)
        self.num_unrestricted = self.n_vars ** 2 - self.n_ones - self.n_nones
        if self.n_ones > self.n_nones:
            self.direction = 'descend'
        else:
            self.direction = 'ascend'

    def _sort_row(self, mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        self.irf_max_full = np.zeros((n_rotation, self.n_vars * self.n_shocks, self.H + 1))

        while counter < n_rotation:
            total += 1
            D = self.draw_rotation()
            self.tool.update(rotation=D)
            _irfs_ = self.tool.irf
            irf_sign = np.sign(np.sum(_irfs_[:, :length_to_check], axis=1).reshape((self.n_vars, self.n_vars)))
            idx, sorted_signs = self._sort_row(irf_sign)
            diff_sign = self.target_signs - sorted_signs

            if np.sum(diff_sign ** 2) == self.num_unrestricted:
                counter += 1
                if verbose:
                    print(f'{counter} accepted rotations/{n_rotation} required rotations')
                D = D[:, idx]
                self.tool.update(rotation=D)
                irfr_full = self.tool.irf
                self.irf_max_full[counter - 1, :, :] = irfr_full[:(self.n_vars ** 2 - self.n_diff * self.n_vars), :]
                irf_needed = irfr_full[:, :h + 1]
                self.irf_mat[counter - 1, :, :] = irf_needed[:(self.n_vars ** 2 - self.n_diff * self.n_vars), :]
                vdr = self.tool.estimate_vd(irfs=irf_needed)
                self.vd_mat[counter - 1, :, :] = vdr[:(self.n_vars ** 2 - self.n_diff * self.n_vars), :]

        # TODO: incorporate HD
        print('*' * 30)
        print(f'acceptance rate is {counter / total}')
