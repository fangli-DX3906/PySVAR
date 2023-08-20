import numpy as np
import palettable.tableau as pt
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Literal

from utils.plot_params import *


class SVARTools:
    def __init__(self,
                 data: np.ndarray,
                 lag_order: int,
                 comp_mat: np.ndarray,
                 cov_mat: np.ndarry,
                 rotation: Optional[np.ndarray] = None):
        self.data = data
        self.lag_order = lag_order
        self.comp_mat = comp_mat
        self.cov_mat = cov_mat
        self.rotation = rotation
        self.n_obs, self.n_vars = self.data.shape
        if rotation is None:
            self.rotation = np.eye(self.n_vars)
        else:
            self.rotation = rotation

    def estimate_irf(self) -> None:
        j = np.concatenate((np.eye(self.n_vars), np.zeros((self.n_vars, self.n_vars * (self.lag_order - 1)))), axis=1)
        aa = np.eye(self.n_vars * self.lag_order)
        # cholesky gives you the lower triangle in numpy
        chol = np.linalg.cholesky(self.cov_mat)
        irf = np.dot(np.dot(np.dot(np.dot(j, aa), j.T), chol), self.rotation)
        irf = irf.reshape((self.n_vars ** 2, -1), order='F')
        for i in range(1, self.n_obs - self.n_vars + 1):
            aa = np.dot(aa, self.comp_mat)
            temp = np.dot(np.dot(np.dot(np.dot(j, aa), j.T), chol), self.rotation)
            temp = temp.reshape((self.n_vars ** 2, -1), order='F')
            irf = np.concatenate((irf, temp), axis=1)
        self.irf = irf

    def estimate_vd(self,
                    irfs: np.ndarray) -> np.ndarray:
        irf_mat = np.transpose(irfs)
        irf_mat_sq = irf_mat ** 2
        irf_mat_sq = irf_mat_sq.reshape((-1, self.n_vars, self.n_vars), order='F')
        irf_sq_sum_h = np.cumsum(irf_mat_sq, axis=0)
        total_fev = np.sum(irf_sq_sum_h, axis=2)
        total_fev_expand = np.expand_dims(total_fev, axis=2)
        vd = irf_sq_sum_h / total_fev_expand
        vd = vd.T.reshape((self.n_vars ** 2, -1))
        return vd

    def estimate_hd(self,
                    shocks: np.ndarray,
                    irfs: np.ndarray) -> np.ndarray:
        hd = np.zeros((self.n_vars, self.n_obs - self.lag_order, self.n_vars))
        for iperiod in range(self.n_obs - self.lag_order):
            for ishock in range(self.n_vars):
                for ivar in range(self.n_vars):
                    shocks_ = shocks[ishock, :iperiod]
                    hd[ivar, iperiod, ishock] = np.dot(irfs[ivar + ishock * self.n_vars, :iperiod], shocks_[::-1])
                    hd = hd.swapaxes(0, 2)
        return hd

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
        self.estimate_irf()

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


class Plottor:
    def __init__(self,
                 var_names: List[str],
                 shock_names: List[str],
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 which_model: Literal['VAR', 'SVAR']):
        self.var_names = var_names
        self.shock_names = shock_names
        self.n_vars = len(var_names)
        self.n_shocks = len(shock_names)
        self.date_frequency = date_frequency
        if which_model == 'VAR':
            prefix = '/orth_shock'

    def plot_irf(self,
                 var_list: List[str],
                 shock_list: List[str],
                 sigs: Union[List[int], int],
                 irf: np.ndarray,
                 with_ci: bool,
                 max_cols: int = 4,
                 irf_cv: Optional[dict] = None,
                 save_path: Optional[str] = None) -> None:

        # define layout
        ns = len(shock_list)
        nv = len(var_list)
        split = nv > max_cols
        n_cols = max_cols if split else nv
        n_rows = nv // max_cols + 1 if split else 1
        h = irf.shape[1]
        x_ticks = range(h)

        # plotting
        for i in range(ns):
            plt.figure(figsize=(n_cols * 10, n_rows * 10))
            plt.subplots_adjust(wspace=0.25, hspace=0.35)
            color = pt.BlueRed_6.mpl_colors[i]
            shock_id = self.shock_names.index(shock_list[i])
            for j in range(nv):
                ax = plt.subplot(n_rows, n_cols, j + 1)
                var_id = self.var_names.index(var_list[j])
                row = var_id + shock_id * self.n_vars
                plt.plot(x_ticks, irf[row, :h + 1], color=color, linewidth=3)
                plt.axhline(y=0, color='black', linestyle='-', linewidth=3)
                if with_ci:
                    for sig, alpha in zip(sigs, alpha_list[1:]):
                        plt.fill_between(x_ticks, irf_cv[sig]['lower'][row, :h + 1], irf_cv[sig]['upper'][row, :h + 1],
                                         alpha=alpha, edgecolor=color, facecolor=color, linewidth=0)
                plt.xlim(0, h)
                plt.xticks(list(range(0, h, 5)))
                plt.title(var_list[j], font_prop_title, pad=5.)
                plt.tick_params(labelsize=25)
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname('Palatino') for label in labels]
                if j == 0:
                    ax.set_xlabel(date_transfer_dict[self.date_frequency], fontdict=font_prop_xlabels, labelpad=1.)
                plt.grid(linestyle='--', linewidth=1.5, color='black', alpha=0.35)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
            plt.suptitle(shock_list[i], fontproperties=font_prop_suptitle)

            # save
            if save_path is not None:
                full_path = save_path + f'/{shock_list[i]}.png'
                plt.savefig(full_path, bbox_inches='tight')
            plt.show()

    def __make_vd_graph(self,
                        var_list: List[str],
                        shock_list: List[str],
                        vd: np.ndarray,
                        max_cols: int = 4,
                        save_path: Optional[str] = None) -> None:
        nv = len(var_list)
        split = nv > max_cols
        n_cols = max_cols if split else nv
        n_rows = nv // max_cols + 1 if split else 1
        h = vd.shape[1]
        x_ticks = range(h)

        plt.figure(figsize=(n_cols * 10, n_rows * 10))
        plt.subplots_adjust(wspace=0.25, hspace=0.35)
        for idxv, var in enumerate(var_list):
            accum = np.zeros(h)
            ax = plt.subplot(n_rows, n_cols, idxv + 1)
            for idxs, sho in enumerate(shock_list):
                color = pt.BlueRed_6.mpl_colors[idxs]
                shock_id = self.shock_names.index(sho)
                var_id = self.var_names.index(var)
                row = var_id + shock_id * self.n_vars
                plt.plot(x_ticks, vd[row, :], color=color, linewidth=3)
                accum += vd[row, :]
                plt.axhline(y=0, color='black', linestyle='-', linewidth=3)
            vd_rest = 1 - accum
            if np.sum(vd_rest) > 1e-10:
                plt.plot(x_ticks, vd_rest, color='k', linewidth=3)
            plt.xlim(0, h - 1)
            plt.xticks(list(range(0, h, 5)))
            plt.title(var, font_prop_title, pad=5.)
            plt.tick_params(labelsize=25)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Palatino') for label in labels]
            if idxv == 0:
                ax.set_xlabel(date_transfer_dict[self.date_frequency], fontdict=font_prop_xlabels, labelpad=1.)
            plt.grid(linestyle='--', linewidth=1.5, color='black', alpha=0.35)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
        plt.suptitle('Variance Decomposition', fontproperties=font_prop_suptitle)

        if save_path is not None:
            full_path = save_path + '/variance_decomposition.png'
            plt.savefig(full_path, bbox_inches='tight')
        plt.show()
