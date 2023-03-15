import datetime
import random
import scipy.io as spio
import pandas as pd
import numpy as np
from time import perf_counter
from typing import Union, Literal, List, Tuple
from SVAR import SetIdentifiedSVARModel


class SignRestriction(SetIdentifiedSVARModel):
    def __init__(self,
                 y: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 target_signs: np.ndarray,
                 data_frequency: Literal['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annually', 'Annually'],
                 date_range: List[datetime.date] = None,  # specific to HD
                 constant: bool = True,
                 criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        '''
        For partial identification, use 1, -1, 0 for the positive, negative, agnostic response.
        '''

        super().__init__(y, var_names, shock_names, data_frequency, date_range, constant, criterion=criterion)
        self.identification = 'sign restriction'
        self.target_signs = target_signs
        self.n_ones = np.sum(self.target_signs == 1)
        self.n_nones = np.sum(self.target_signs == -1)
        self.num_unrestricted = self.n_vars ** 2 - self.n_ones - self.n_nones
        if self.n_ones > self.n_nones:
            self.direction = 'descend'
        else:
            self.direction = 'ascend'

    def sort_row(self,
                 mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def get_rotation(self) -> np.ndarray:
        raw_mat = np.random.randn(self.n_vars, self.n_vars)
        Q, R = np.linalg.qr(raw_mat)
        Q = np.sign(np.diag(R)).reshape((-1, 1)) * Q

        return Q

    def identify(self,
                 h: int,
                 n_rotation: int,
                 irf_sig: Union[List[int], int],
                 length_to_check: int = 1,
                 with_vd: bool = False,
                 vd_sig: Union[List[int], int, None] = None,
                 seed: Union[bool, int] = False,
                 verbose: bool = False):
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        counter = 0
        self.irf_mat = np.zeros((n_rotation, self.n_vars * self.n_shocks, h + 1))

        if with_vd:
            self.vd_mat = np.zeros((n_rotation, self.n_vars * self.n_shocks, h + 1))
        else:
            self.vd_mat = None

        while counter < n_rotation:
            D = self.get_rotation()
            irfs = self.get_irf(h=h, rotation=D)
            irf_sign = np.sign(np.sum(irfs[:, :length_to_check], axis=1).reshape((self.n_vars, self.n_vars)))

            idx, sorted_signs = self.sort_row(irf_sign)
            diff_sign = self.target_signs - sorted_signs

            if np.sum(diff_sign ** 2) == self.num_unrestricted:
                counter += 1
                if verbose:
                    print(f'{counter} accepted rotation/{n_rotation} required rotations')

                D = D[:, idx]
                irfs = self.get_irf(h=h, rotation=D)
                self.irf_mat[counter - 1, :, :] = irfs[:(self.n_vars ** 2 - self.n_diff * self.n_vars), :]

                if with_vd:
                    vds = self.get_vd(h=h, irf_data=irfs)
                    self.vd_mat[counter - 1, :, :] = vds[:(self.n_vars ** 2 - self.n_diff * self.n_vars), :]

        self.pack_up_irf(irf_sig=irf_sig, irf_mat=self.irf_mat, median_as_point_estimate=True)
        if with_vd:
            self.pack_up_vd(vd_sig=vd_sig, vd_mat=self.vd_mat, median_as_point_estimate=True)


if __name__ == '__main__':
    # we simulate the data (using the directional information) and data does not make sense, just for the exercises.
    data = pd.read_csv('data/data1.csv')
    data = np.array(data)[:, 1:]
    names = ['var1', 'var2', 'var3', 'var4']
    shocks = ['sh1', 'sh2', 'sh3']
    signs = np.zeros((len(names), len(names)))
    signs[0, :] = np.array([1, 0, 1, 0])
    signs[1, :] = np.array([1, 0, -1, 0])
    signs[2, :] = np.array([-1, 0, 0, 0])
    fsr = SignRestriction(var_names=names, shock_names=shocks, target_signs=signs, y=data,
                          data_frequency='Quarterly')
    fsr.identify(h=20, n_rotation=50, irf_sig=[68, 80, 95], vd_sig=[68, 80, 95], with_vd=True, seed=True, verbose=True)
    fsr.plot_irf(var_list=['var1', 'var2', 'var3'], shock_list=['sh1', 'sh2', 'sh3'], sigs=[68, 95], with_ci=True)
    fsr.plot_vd(var_list=['var1', 'var2', 'var3'], shock_list=['sh1', 'sh2', 'sh3'])
