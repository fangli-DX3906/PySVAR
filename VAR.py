import numpy as np
import palettable.tableau as pt
import scipy.io as spio
import random
import matplotlib.pyplot as plt
import itertools
import datetime
from typing import Union, Literal, Tuple, List, Optional
from BasicVAR import BasicVARModel


class VARModel(BasicVARModel):
    def __init__(self,
                 y: np.ndarray,
                 var_names: list,
                 data_frequency: Literal['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annually', 'Annually'],
                 date_range: List[datetime.date] = None,  # specific to HD
                 constant: bool = True):
        super().__init__(y, var_names, data_frequency, date_range, constant)

    # TODO: finish this, second stage: move this to the base class!
    def __repr__(self):
        pass

    def irf(self,
            h: int,
            comp_mat: Optional[np.ndarray] = None,
            cov_mat: Optional[np.ndarray] = None) -> np.ndarray:

        if comp_mat is None:
            comp_mat = self.comp_mat
        if cov_mat is None:
            cov_mat = self.cov_mat

        # according to Kilian p109
        j = np.concatenate((np.eye(self.n_vars), np.zeros((self.n_vars, self.n_vars * (self.lag_order - 1)))), axis=1)
        aa = np.eye(self.n_vars * self.lag_order)
        chol = np.linalg.cholesky(cov_mat)  # cholesky gives you the lower triangle in numpy
        irf = np.dot(np.dot(np.dot(j, aa), j.T), chol)
        irf = irf.reshape((self.n_vars ** 2, -1), order='F')

        for i in range(1, h + 1):
            aa = np.dot(aa, comp_mat)
            temp = np.dot(np.dot(np.dot(j, aa), j.T), chol)
            temp = temp.reshape((self.n_vars ** 2, -1), order='F')
            irf = np.concatenate((irf, temp), axis=1)

        self.irf_point_estimate = irf
        return irf

    def vd(self,
           h: int,
           irf_mat: Optional[np.ndarray] = None,
           comp_mat: Optional[np.ndarray] = None,
           cov_mat: Optional[np.ndarray] = None) -> np.ndarray:

        if irf_mat is None:
            irf_mat = np.transpose(self.irf(h, comp_mat, cov_mat))
        else:
            irf_mat = np.transpose(irf_mat)

        irf_mat_sq = irf_mat ** 2
        irf_mat_sq = irf_mat_sq.reshape((-1, self.n_vars, self.n_vars), order='F')
        irf_sq_sum_h = np.cumsum(irf_mat_sq, axis=0)
        total_fev = np.sum(irf_sq_sum_h, axis=2)
        total_fev_expand = np.expand_dims(total_fev, axis=2)
        vd = irf_sq_sum_h / total_fev_expand
        vd = vd.T.reshape((self.n_vars ** 2, -1))
        self.vd_point_estimate = vd

        return vd

    def boot_confid_intvl(self,
                          h: int,
                          n_path: int,
                          sigs_irf: Union[List[int], int],
                          sigs_vd: Union[List[int], int, None] = None,
                          with_vd: bool = False,
                          seed: Union[bool, int] = False,
                          verbose: bool = False):
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.irf_mat = np.zeros((n_path, self.n_vars ** 2, h + 1))
        if with_vd:
            self.vd_mat = np.zeros((n_path, self.n_vars ** 2, h + 1))
        else:
            self.vd_mat = None

        y_data = self.data.T
        yy = y_data[:, self.lag_order - 1:self.n_obs]
        for i in range(1, self.lag_order):
            yy = np.concatenate((yy, y_data[:, self.lag_order - i - 1:self.n_obs - i]), axis=0)

        yyr = np.zeros((self.lag_order * self.n_vars, self.n_obs - self.lag_order + 1))
        u = self.resids
        index_set = range(self.n_obs - self.lag_order)
        for r in range(n_path):
            if verbose:
                print(f'This is {r + 1} simulation.')

            pos = random.randint(0, self.n_obs - self.lag_order)
            yyr[:, 0] = yy[:, pos]
            idx = np.random.choice(index_set, size=self.n_obs - self.lag_order)
            ur = np.concatenate((np.zeros((self.lag_order * self.n_vars, 1)), u[:, idx]), axis=1)
            for i in range(1, self.n_obs - self.lag_order + 1):
                yyr[:, i] = self._intercepts.T + np.dot(self.comp_mat, yyr[:, i - 1]) + ur[:, i]

            yr = yyr[:self.n_vars, :]
            for i in range(1, self.lag_order):
                temp = yyr[i * self.n_vars:(i + 1) * self.n_vars, 0].reshape((-1, 1))
                yr = np.concatenate((temp, yr), axis=1)

            yr = yr.T
            comp_mat_r, cov_mat_r, _, _, _ = self._fit(yr, self.lag_order)
            irfr = self.irf(h, comp_mat=comp_mat_r, cov_mat=cov_mat_r[:self.n_vars, :self.n_vars])
            self.irf_mat[r, :, :] = irfr
            if with_vd:
                vdr = self.vd(h, irf_mat=irfr)
                self.vd_mat[r, :, :] = vdr

        self.pack_up(h=h, sig_irf=sigs_irf, irfs=self.irf_mat, vds=self.vd_mat, sig_vd=sigs_vd)

    def pack_up(self,
                h: int,
                sig_irf: Union[List[int], int, None],
                irfs: Optional[np.ndarray],
                vds: Optional[np.ndarray],
                sig_vd: Union[List[int], int, None]):  # for the set identified method such as sign restriction

        self.h = h
        self.irf_confid_intvl = dict()
        if not isinstance(sig_irf, list):
            sig_irf = [sig_irf]

        for sig in sig_irf:
            self.irf_confid_intvl[sig] = {}
            self.irf_confid_intvl[sig]['lower'] = np.percentile(irfs, (100 - sig) / 2, axis=0)
            self.irf_confid_intvl[sig]['upper'] = np.percentile(irfs, 100 - (100 - sig) / 2, axis=0)

        if vds is not None:
            self.vd_confid_intvl = dict()
            if not isinstance(sig_vd, list):
                sig_vd = [sig_vd]

            for sig in sig_vd:
                self.vd_confid_intvl[sig] = {}
                self.vd_confid_intvl[sig]['lower'] = np.percentile(vds, (100 - sig) / 2, axis=0)
                self.vd_confid_intvl[sig]['upper'] = np.percentile(vds, 100 - (100 - sig) / 2, axis=0)

    def plot_irf(self,
                 var_list: Optional[List[str]] = None,
                 shock_list: Optional[List[int]] = None,
                 sigs: Union[List[int], int, None] = None,
                 with_ci: bool = True):

        if var_list is None:
            var_list = self.var_names
        if shock_list is None:
            shock_list = list(range(self.n_vars))

        row_idx = []
        for j in shock_list:
            for var in var_list:
                idx = self.var_names.index(var)
                row_idx.append(idx + j * self.n_vars)

        if not isinstance(sigs, list):
            sigs = [sigs]

        alpha_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
        font_prop_xlabels = {'family': 'Times New Roman',
                             'size': 30}
        font_prop_ylabels = {'family': 'Times New Roman',
                             'size': 40}
        font_prop_title = {'family': 'Times New Roman',
                           'size': 40}

        n_rows = len(shock_list)
        n_cols = len(var_list)
        x_ticks = range(self.h + 1)
        plt.figure(figsize=(n_cols * 10, n_rows * 10))
        plt.subplots_adjust(wspace=0.25, hspace=0.35)

        for (i, j), row in zip(itertools.product(range(n_rows), range(n_cols)), row_idx):
            color = pt.BlueRed_6.mpl_colors[i]
            ax = plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
            plt.plot(x_ticks, self.irf_point_estimate[row, :], color=color, linewidth=3)
            plt.axhline(y=0, color='black', linestyle='-', linewidth=3)
            if with_ci:
                for sig, alpha in zip(sigs, alpha_list[1:]):
                    plt.fill_between(x_ticks,
                                     self.irf_confid_intvl[sig]['lower'][row, :],
                                     self.irf_confid_intvl[sig]['upper'][row, :],
                                     alpha=alpha, edgecolor=color, facecolor=color, linewidth=0)
            plt.xlim(0, self.h)
            plt.xticks(list(range(0, self.h + 1, 5)))
            plt.title(var_list[j], font_prop_title, pad=5.)
            plt.tick_params(labelsize=25)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Palatino') for label in labels]
            if i == 0 and j == 0:
                ax.set_xlabel(self.data_frequency, fontdict=font_prop_xlabels, labelpad=1.)
            if j == 0:
                ax.set_ylabel(f'orthogonal shock {i + 1}', fontdict=font_prop_ylabels, labelpad=25.)
            plt.grid(linestyle='--', linewidth=1.5, color='black', alpha=0.35)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

        plt.show()

    # TODO!!!
    def plot_vd(self):
        pass


if __name__ == '__main__':
    # data = ...
    # data = data[:100, [0, 6]]
    # names = ['Output', 'Inflation']
    # var = VARModel(y=data, var_names=names, data_frequency='Quarterly')
    # var.fit()

    # irf = var.irf(h=20)
    # vd = var.vd(h=20)
    # var.boot_confid_intvl(h=20, n_path=1000, sigs_irf=[68, 80, 95], with_vd=False, verbose=True, seed=3906)
    # var.plot_irf(var_list=names, shock_list=[0], sigs=[68, 80], with_ci=True)
