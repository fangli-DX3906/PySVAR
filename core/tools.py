import numpy as np
from typing import Optional, Union, List


class Tools:
    def __init__(self,
                 data: np.ndarray,
                 lag_order: int,
                 comp_mat: np.ndarray,
                 cov_mat: np.ndarray,
                 rotation: Optional[np.ndarray] = None):

        self.data = data
        self.lag_order = lag_order
        self.comp_mat = comp_mat
        self.cov_mat = cov_mat
        self.rotation = rotation
        self.n_obs, self.n_vars = self.data.shape

        if rotation is not None:
            self.rotation = rotation
            self.reduced_var_irf_point_estimate = self.estimate_irf()

    def estimate_irf(self, length: Optional[int] = None) -> np.ndarray:
        j = np.concatenate((np.eye(self.n_vars), np.zeros((self.n_vars, self.n_vars * (self.lag_order - 1)))), axis=1)
        aa = np.eye(self.n_vars * self.lag_order)
        chol = np.linalg.cholesky(self.cov_mat)  # lower triangle
        irf = np.dot(np.dot(np.dot(np.dot(j, aa), j.T), chol), self.rotation)
        irf = irf.reshape((self.n_vars ** 2, -1), order='F')

        if length is not None:
            H = length
        else:
            H = self.n_obs - self.lag_order + 1
        for i in range(1, H):
            aa = np.dot(aa, self.comp_mat)
            temp = np.dot(np.dot(np.dot(np.dot(j, aa), j.T), chol), self.rotation)
            temp = temp.reshape((self.n_vars ** 2, -1), order='F')
            irf = np.concatenate((irf, temp), axis=1)

        return irf

    def estimate_vd(self, irfs: np.ndarray) -> np.ndarray:
        irf_mat = np.transpose(irfs)
        irf_mat_sq = irf_mat ** 2
        irf_mat_sq = irf_mat_sq.reshape((-1, self.n_vars, self.n_vars), order='F')
        irf_sq_sum_h = np.cumsum(irf_mat_sq, axis=0)
        total_fev = np.sum(irf_sq_sum_h, axis=2)
        total_fev_expand = np.expand_dims(total_fev, axis=2)
        vd = irf_sq_sum_h / total_fev_expand
        vd = vd.T.reshape((self.n_vars ** 2, -1))

        return vd

    def estimate_hd(self, shocks: np.ndarray, irfs: np.ndarray) -> np.ndarray:
        hd = np.zeros((self.n_vars ** 2, self.n_obs - self.lag_order))

        for s in range(self.n_vars):
            for v in range(self.n_vars):
                for l in range(self.n_obs - self.lag_order):
                    stemp = shocks[s, :l + 1]
                    hd[s * self.n_vars + v, l] = np.dot(irfs[s * self.n_vars + v, :l + 1], stemp[::-1])

        return hd

    def make_confid_intvl(self, mat: np.ndarray, length: Optional[int], sigs: Union[int, List[int]]) -> dict:
        if length is not None:
            mat = mat[:, :, :length]

        confid_intvl = dict()
        if not isinstance(sigs, list):
            sigs = [sigs]

        for sig in sigs:
            confid_intvl[sig] = dict()
            confid_intvl[sig]['lower'] = np.percentile(mat, (100 - sig) / 2, axis=0)
            confid_intvl[sig]['upper'] = np.percentile(mat, 100 - (100 - sig) / 2, axis=0)

        return confid_intvl

    def update(self, **kwargs) -> None:
        to_be_updated = list(kwargs.keys())

        if 'comp' in to_be_updated:
            self.comp_mat = kwargs['comp']
        if 'cov' in to_be_updated:
            self.cov_mat = kwargs['cov']
        if 'lag' in to_be_updated:
            self.lag_order = kwargs['lag']
        if 'data' in to_be_updated:
            self.data = kwargs['data']
        if 'rotation' in to_be_updated:
            self.rotation = kwargs['rotation']
