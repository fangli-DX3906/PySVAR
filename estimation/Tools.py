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
        self.lag_order_ = lag_order
        self.comp_mat = comp_mat
        self.cov_mat = cov_mat
        self.rotation = rotation
        self.n_obs_, self.n_vars_ = self.data.shape
        if rotation is None:
            self.rotation = np.eye(self.n_vars_)
        else:
            self.rotation = rotation
        self.estimate_irf()
        # self._irfs_ is the point estimate, self.irf varies with each update
        self._irfs_ = self.irf

    def estimate_irf(self) -> None:
        j = np.concatenate((np.eye(self.n_vars_), np.zeros((self.n_vars_, self.n_vars_ * (self.lag_order_ - 1)))),
                           axis=1)
        aa = np.eye(self.n_vars_ * self.lag_order_)
        # cholesky gives you the lower triangle in numpy
        chol = np.linalg.cholesky(self.cov_mat)
        irf = np.dot(np.dot(np.dot(np.dot(j, aa), j.T), chol), self.rotation)
        irf = irf.reshape((self.n_vars_ ** 2, -1), order='F')
        for i in range(1, self.n_obs_ - self.lag_order_ + 1):
            aa = np.dot(aa, self.comp_mat)
            temp = np.dot(np.dot(np.dot(np.dot(j, aa), j.T), chol), self.rotation)
            temp = temp.reshape((self.n_vars_ ** 2, -1), order='F')
            irf = np.concatenate((irf, temp), axis=1)
        self.irf = irf

    def estimate_vd(self,
                    irfs: np.ndarray) -> np.ndarray:
        irf_mat = np.transpose(irfs)
        irf_mat_sq = irf_mat ** 2
        irf_mat_sq = irf_mat_sq.reshape((-1, self.n_vars_, self.n_vars_), order='F')
        irf_sq_sum_h = np.cumsum(irf_mat_sq, axis=0)
        total_fev = np.sum(irf_sq_sum_h, axis=2)
        total_fev_expand = np.expand_dims(total_fev, axis=2)
        vd = irf_sq_sum_h / total_fev_expand
        vd = vd.T.reshape((self.n_vars_ ** 2, -1))
        return vd

    def estimate_hd(self,
                    shocks: np.ndarray,
                    irfs: np.ndarray) -> np.ndarray:
        hd = np.zeros((self.n_vars_, self.n_obs_ - self.lag_order, self.n_vars_))
        for iperiod in range(self.n_obs_ - self.lag_order):
            for ishock in range(self.n_vars_):
                for ivar in range(self.n_vars_):
                    shocks_ = shocks[ishock, :iperiod]
                    hd[ivar, iperiod, ishock] = np.dot(irfs[ivar + ishock * self.n_vars_, :iperiod], shocks_[::-1])
                    hd = hd.swapaxes(0, 2)
        return hd

    def make_confid_intvl(self,
                          mat: np.ndarray,
                          sigs: Union[int, List]) -> dict:
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
        self.estimate_irf()
