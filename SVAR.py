import datetime
import random
import palettable.tableau as pt
from typing import Union, Literal, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import datetime
import itertools
from BasicVAR import BasicVARModel


class SVARModel(BasicVARModel):
    def __init__(self,
                 y: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 data_frequency: Literal['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annually', 'Annually'],
                 date_range: List[datetime.date] = None,  # specific to HD
                 constant: bool = True,
                 criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(y, var_names, data_frequency, date_range, constant)
        self.shock_names = shock_names
        self.n_shocks = len(shock_names)
        self.n_diff = self.n_vars - self.n_shocks
        self.fit(criterion=criterion)
        self.chol = np.linalg.cholesky(self.cov_mat)  # cholesky gives you the lower triangle in numpy

    def get_irf(self,
                h: int,
                rotation: np.ndarray,
                comp_mat: np.ndarray,
                cov_mat: np.ndarray) -> np.ndarray:

        chol = np.linalg.cholesky(cov_mat)

        # according to Kilian p109
        j = np.concatenate((np.eye(self.n_vars), np.zeros((self.n_vars, self.n_vars * (self.lag_order - 1)))), axis=1)
        aa = np.eye(self.n_vars * self.lag_order)
        irf = np.dot(np.dot(np.dot(np.dot(j, aa), j.T), chol), rotation)
        irf = irf.reshape((self.n_vars ** 2, -1), order='F')

        for i in range(1, h + 1):
            aa = np.dot(aa, comp_mat)
            temp = np.dot(np.dot(np.dot(np.dot(j, aa), j.T), chol), rotation)
            temp = temp.reshape((self.n_vars ** 2, -1), order='F')
            irf = np.concatenate((irf, temp), axis=1)

        return irf

    def get_vd(self,
               h: int,
               irf_data: np.ndarray) -> np.ndarray:

        if h > irf_data.shape[1] - 1:
            h = irf_data.shape[1] - 1

        irf_data = np.transpose(irf_data)
        irf_mat_sq = irf_data ** 2
        irf_mat_sq = irf_mat_sq.reshape((-1, self.n_vars, self.n_vars), order='F')
        irf_sq_sum_h = np.cumsum(irf_mat_sq, axis=0)
        total_fev = np.sum(irf_sq_sum_h, axis=2)
        total_fev_expand = np.expand_dims(total_fev, axis=2)
        vd = irf_sq_sum_h / total_fev_expand
        vd = vd.T.reshape((self.n_vars ** 2, -1))
        # TODO：h=?
        vd = vd[:, :h + 1]

        return vd

    def get_hd(self):
        pass

    def get_structural_shocks(self,
                              rotation: np.ndarray) -> np.ndarray:
        return np.dot(np.linalg.inv(np.dot(self.chol, rotation)), self.resids[:self.n_vars, :]).T

    # TODO: finish this method.
    def set_params(self):
        pass

    def pack_up_irf(self,
                    irf_mat: np.ndarray,
                    irf_sig: Union[List[int], int],
                    median_as_point_estimate: bool):

        if irf_mat is None:
            raise ValueError('None value!')

        self.irf_confid_intvl = dict()
        if not isinstance(irf_sig, list):
            irf_sig = [irf_sig]

        for sig in irf_sig:
            self.irf_confid_intvl[sig] = {}
            self.irf_confid_intvl[sig]['lower'] = np.percentile(irf_mat, (100 - sig) / 2, axis=0)
            self.irf_confid_intvl[sig]['upper'] = np.percentile(irf_mat, 100 - (100 - sig) / 2, axis=0)

        if median_as_point_estimate:
            self.irf_point_estimate = np.percentile(irf_mat, 50, axis=0)

    def pack_up_vd(self,
                   vd_mat: np.ndarray,
                   vd_sig: Union[List[int], int],
                   median_as_point_estimate: bool):

        if vd_mat is None:
            raise ValueError('None value!')

        self.vd_confid_intvl = dict()
        if not isinstance(vd_sig, list):
            vd_sig = [vd_sig]

        for sig in vd_sig:
            self.vd_confid_intvl[sig] = {}
            self.vd_confid_intvl[sig]['lower'] = np.percentile(vd_mat, (100 - sig) / 2, axis=0)
            self.vd_confid_intvl[sig]['upper'] = np.percentile(vd_mat, 100 - (100 - sig) / 2, axis=0)

        if median_as_point_estimate:
            self.vd_point_estimate = np.percentile(vd_mat, 50, axis=0)

    def plot_irf(self,
                 h: Optional[int] = None,
                 var_list: Optional[List] = None,
                 shock_list: Optional[List] = None,
                 sigs: Union[List[int], int, None] = None,
                 irf_confid_intvl: Optional[dict] = None,
                 irf_point_estimate: Optional[np.ndarray] = None,
                 with_ci: bool = True):

        if irf_confid_intvl is None:
            irf_confid_intvl = self.irf_confid_intvl
        if irf_point_estimate is None:
            irf_point_estimate = self.irf_point_estimate

        if h is None:
            h = irf_point_estimate.shape[1] - 1
        else:
            if h > irf_point_estimate.shape[1] - 1:
                h = irf_point_estimate.shape[1] - 1

        row_idx = []
        for shock in shock_list:
            try:
                j = self.shock_names.index(shock)
            except:
                raise ValueError(f'{shock} is not initialized!')
            for var in var_list:
                try:
                    i = self.var_names.index(var)
                except:
                    raise ValueError(f'{var} is not in your system!')
                row_idx.append(i + j * self.n_vars)

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
        x_ticks = range(h + 1)
        plt.figure(figsize=(n_cols * 10, n_rows * 10))
        plt.subplots_adjust(wspace=0.25, hspace=0.35)

        for (i, j), row in zip(itertools.product(range(n_rows), range(n_cols)), row_idx):
            color = pt.BlueRed_6.mpl_colors[i]
            ax = plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
            plt.plot(x_ticks, irf_point_estimate[row, :], color=color, linewidth=3)
            plt.axhline(y=0, color='black', linestyle='-', linewidth=3)
            if with_ci:
                for sig, alpha in zip(sigs, alpha_list[1:]):
                    plt.fill_between(x_ticks,
                                     irf_confid_intvl[sig]['lower'][row, :],
                                     irf_confid_intvl[sig]['upper'][row, :],
                                     alpha=alpha, edgecolor=color, facecolor=color, linewidth=0)
            plt.xlim(0, h)
            plt.xticks(list(range(0, h + 1, 5)))
            plt.title(var_list[j], font_prop_title, pad=5.)
            plt.tick_params(labelsize=25)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Palatino') for label in labels]
            if i == 0 and j == 0:
                ax.set_xlabel(self.data_frequency, fontdict=font_prop_xlabels, labelpad=1.)
            if j == 0:
                ax.set_ylabel(shock_list[i], fontdict=font_prop_ylabels, labelpad=25.)
            plt.grid(linestyle='--', linewidth=1.5, color='black', alpha=0.35)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

        plt.show()

    # TODO: how to add the confid intvl
    def plot_vd(self,
                h: Optional[int] = None,
                var_list: Optional[List] = None,
                shock_list: Optional[List] = None,
                vd_point_estimate: Optional[np.ndarray] = None):
        # add_rest_shocks: bool = True,
        # with_ci: bool = False,
        # sigs: Union[List[int], int, None] = None,
        # vd_confid_intvl: Optional[dict] = None):

        if vd_point_estimate is None:
            vd_point_estimate = self.vd_point_estimate
        # if vd_confid_intvl is None:
        #     vd_confid_intvl = self.vd_confid_intvl

        if h is None:
            h = vd_point_estimate.shape[1] - 1
        else:
            if h > vd_point_estimate.shape[1] - 1:
                h = vd_point_estimate.shape[1] - 1

        # if not isinstance(sigs, list):
        #     sigs = [sigs]
        # alpha_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

        font_prop_xlabels = {'family': 'Times New Roman',
                             'size': 30}
        # font_prop_ylabels = {'family': 'Times New Roman',
        #                      'size': 40}
        font_prop_title = {'family': 'Times New Roman',
                           'size': 40}

        n_cols = len(var_list)
        x_ticks = range(h + 1)
        plt.figure(figsize=(n_cols * 10, 10))
        plt.subplots_adjust(wspace=0.25, hspace=0.35)

        for idxv, var in enumerate(var_list):
            accum = np.zeros(h + 1)
            ax = plt.subplot(1, n_cols, idxv + 1)
            for idxs, sho in enumerate(shock_list):
                color = pt.BlueRed_6.mpl_colors[idxs]
                j = self.shock_names.index(sho)
                i = self.var_names.index(var)
                row = i + j * self.n_vars
                plt.plot(x_ticks, vd_point_estimate[row, :], color=color, linewidth=3)
                accum += vd_point_estimate[row, :]
                plt.axhline(y=0, color='black', linestyle='-', linewidth=3)
            vd_rest = 1 - accum
            plt.plot(x_ticks, vd_rest, color='k', linewidth=3)
            plt.xlim(0, h)
            plt.xticks(list(range(0, h + 1, 5)))
            plt.title(var, font_prop_title, pad=5.)
            plt.tick_params(labelsize=25)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Palatino') for label in labels]
            if idxv == 0:
                ax.set_xlabel(self.data_frequency, fontdict=font_prop_xlabels, labelpad=1.)
            plt.grid(linestyle='--', linewidth=1.5, color='black', alpha=0.35)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
        plt.show()

    def plot_hd(self):
        pass


class SetIdentifiedSVARModel(SVARModel):
    def __init__(self,
                 y: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 data_frequency: Literal['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annually', 'Annually'],
                 date_range: List[datetime.date] = None,
                 constant: bool = True,
                 criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(y, var_names, shock_names, data_frequency, date_range, constant, criterion=criterion)
        self.irf_mat = None
        self.vd_mat = None

    def get_irf(self,
                h: int,
                rotation: np.ndarray,
                comp_mat: Optional[np.ndarray] = None,
                cov_mat: Optional[np.ndarray] = None) -> np.ndarray:

        if comp_mat is None:
            comp_mat = self.comp_mat
        if cov_mat is None:
            cov_mat = self.cov_mat
        return super().get_irf(h=h, rotation=rotation, comp_mat=comp_mat, cov_mat=cov_mat)


class PointIdentifiedSVARModel(SVARModel):
    def __init__(self,
                 y: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 data_frequency: Literal['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annually', 'Annually'],
                 date_range: List[datetime.date] = None,
                 constant: bool = True,
                 criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(y, var_names, shock_names, data_frequency, date_range, constant, criterion=criterion)
        self.rotation = None

    def get_irf(self,
                h: int,
                rotation: Optional[np.ndarray] = None,
                comp_mat: Optional[np.ndarray] = None,
                cov_mat: Optional[np.ndarray] = None) -> np.ndarray:

        if comp_mat is None:
            comp_mat = self.comp_mat
        if cov_mat is None:
            cov_mat = self.cov_mat
        if rotation is None:
            rotation = self.rotation

        return super().get_irf(h=h, rotation=rotation, comp_mat=comp_mat, cov_mat=cov_mat)

    def bootstrap(self,
                  h: int,
                  n_path: int,
                  with_vd: bool = True,
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
            cov_mat_r = cov_mat_r[:self.n_vars, :self.n_vars]
            rotationr = self.identify(comp_mat=comp_mat_r, cov_mat=cov_mat_r)
            irfr = self.get_irf(h, rotation=rotationr, comp_mat=comp_mat_r, cov_mat=cov_mat_r)
            self.irf_mat[r, :, :] = irfr
            if with_vd:
                vdr = self.get_vd(h, irf_data=irfr)
                self.vd_mat[r, :, :] = vdr
