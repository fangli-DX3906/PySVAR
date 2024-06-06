import numpy as np
import palettable.lightbartlein as pl
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Literal

plt.rcParams['text.usetex'] = True
plt.rcParams['font.serif'] = ['Palatino']
plt.rcParams['font.family'] = 'serif'
color_set = [pl.diverging.BlueOrange12_2.mpl_colors[0],
             pl.diverging.BlueOrange12_2.mpl_colors[1],
             pl.diverging.BrownBlue10_2.mpl_colors[0],
             pl.diverging.BrownBlue10_2.mpl_colors[1],
             pl.diverging.GreenMagenta_2.mpl_colors[0],
             pl.diverging.GreenMagenta_2.mpl_colors[1]]
alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9]


class Plotting:
    def __init__(self,
                 var_names: List[str],
                 shock_names: List[str],
                 date_frequency: Literal['M', 'Q', 'A']):

        self.var_names = var_names
        self.shock_names = shock_names
        self.n_vars = len(var_names)
        self.n_shocks = len(shock_names)
        self.date_frequency = date_frequency
        self.dict = {'M': 'Monthly', 'Q': 'Quarterly', 'A': 'Annually'}

    def plot_irf(self,
                 h: int,
                 var_list: List[str],
                 shock_list: List[str],
                 sigs: Union[List[int], int],
                 irf: np.ndarray,
                 with_ci: bool,
                 irf_ci: Optional[dict],
                 save_path: Optional[str]) -> None:

        # define layout
        ns = len(shock_list)
        nv = len(var_list)
        if nv <= 3:
            n_cols = nv
            n_rows = 1
            legend_pos = (-0.8, -0.15)
        elif nv == 4:
            n_cols = 2
            n_rows = 2
            legend_pos = (-0.15, -0.15)
        else:
            n_cols = 3
            n_rows = int(nv / 3) if nv % 3 == 0 else nv // 3 + 1
            legend_pos = (-0.8, -0.15)
        x_ticks = range(h + 1)

        for i_shock in range(ns):
            plt.figure(figsize=(n_cols * 10, n_rows * 9))
            plt.subplots_adjust(wspace=0.23, hspace=0.37, left=0.07, right=0.95, top=0.93, bottom=0.2)
            shock_id = self.shock_names.index(shock_list[i_shock])
            shock_name = self.shock_names[shock_id]
            color = color_set[i_shock]

            for i_var in range(nv):
                ax = plt.subplot(n_rows, n_cols, i_var + 1)
                var_id = self.var_names.index(var_list[i_var])
                row = var_id + shock_id * self.n_vars
                plt.plot(x_ticks, irf[row, :], color=color, linewidth=6)
                plt.title(self.var_names[var_id], {'size': 37}, pad=5.)
                if with_ci:
                    for sig, alpha in zip(sigs, alpha_list[:len(sigs)]):
                        plt.fill_between(x_ticks,
                                         irf_ci[sig]['lower'][row, :],
                                         irf_ci[sig]['upper'][row, :],
                                         label=rf'${sig}\%$ Confidence Interval',
                                         alpha=alpha, edgecolor=color, facecolor=color, linewidth=0)
                if i_var == 0:
                    ax.set_xlabel(self.dict[self.date_frequency], {'size': 27}, labelpad=15.)
                    ax.set_ylabel('Percentage', {'size': 27}, labelpad=17.)

                plt.axhline(y=0, color='black', linestyle='-')
                plt.xlim(0, h)
                plt.xticks(list(range(0, h, 5)))
                plt.tick_params(labelsize=27)
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname('Palatino') for label in labels]
                plt.grid(linewidth=1, color='black', alpha=0.1)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_linewidth(1)
                ax.spines['bottom'].set_linewidth(1)

            plt.legend(prop={'size': 35}, loc='upper center', bbox_to_anchor=legend_pos, ncol=len(sigs) + 1,
                       frameon=False, borderaxespad=-0.1)
            if save_path is not None:
                plt.savefig(save_path + f'/irf_{shock_name}.png', bbox_inches='tight', dpi=200)
            plt.show()

    def plot_vd(self,
                h: int,
                var_list: List[str],
                shock_list: List[str],
                vd: np.ndarray,
                save_path: Optional[str] = None) -> None:

        nv = len(var_list)
        if nv <= 3:
            n_cols = nv
            n_rows = 1
            legend_pos = (-0.8, -0.15)
        elif nv == 4:
            n_cols = 2
            n_rows = 2
            legend_pos = (-0.15, -0.15)
        else:
            n_cols = 3
            n_rows = nv // 3 + 1
            legend_pos = (-0.8, -0.15)
        x_ticks = range(h + 1)

        plt.figure(figsize=(n_cols * 10, n_rows * 9))
        plt.subplots_adjust(wspace=0.23, hspace=0.37, left=0.07, right=0.95, top=0.93, bottom=0.2)

        for i_var, var in enumerate(var_list):
            ax = plt.subplot(n_rows, n_cols, i_var + 1)
            accum = np.zeros(h + 1)

            for i_shock, shock in enumerate(shock_list):
                color = color_set[i_shock]
                shock_id = self.shock_names.index(shock)
                var_id = self.var_names.index(var)
                row = var_id + shock_id * self.n_vars
                plt.plot(x_ticks, vd[row, :], color=color, linewidth=6, label=shock)
                accum += vd[row, :]
                plt.axhline(y=0, color='black', linestyle='-')

            vd_rest = 1 - accum
            if np.sum(vd_rest) > 1e-10:
                plt.plot(x_ticks, vd_rest, color='black', linewidth=6)

            plt.title(self.var_names[var_id], {'size': 37}, pad=5.)

            if i_var == 0:
                ax.set_xlabel(self.dict[self.date_frequency], {'size': 27}, labelpad=15.)
                ax.set_ylabel('Percentage', {'size': 27}, labelpad=17.)

            plt.xlim(0, h)
            plt.xticks(list(range(0, h, 5)))
            plt.tick_params(labelsize=27)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Palatino') for label in labels]
            plt.grid(linewidth=1, color='black', alpha=0.1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)

        plt.legend(prop={'size': 35}, loc='upper center', bbox_to_anchor=legend_pos, ncol=3,
                   frameon=False, borderaxespad=-0.1)

        if save_path is not None:
            plt.savefig(save_path + f'/vd.png', bbox_inches='tight', dpi=200)
        plt.show()

    def plot_hd(self,
                var_name: str,
                shock_name: str,
                date_list: list,
                hd: np.ndarray,
                with_ci: bool,
                sigs: Union[List[int], int],
                hd_ci: Optional[dict],
                save_path: Optional[str] = None) -> None:

        plt.figure(figsize=(18, 5))
        plt.subplots_adjust(hspace=0.5, left=0.07, right=0.95, top=0.93, bottom=0.1)
        ax = plt.subplot(1, 1, 1)
        shock_id = self.shock_names.index(shock_name)
        var_id = self.var_names.index(var_name)
        row = var_id + shock_id * self.n_vars
        x_ticks = range(hd.shape[1])
        color = color_set[0]
        plt.plot(date_list, hd[row, :], color=color, linewidth=4)
        if with_ci:
            for sig, alpha in zip(sigs, alpha_list[:len(sigs)]):
                plt.fill_between(x_ticks,
                                 hd_ci[sig]['lower'][row, :],
                                 hd_ci[sig]['upper'][row, :],
                                 label=rf'${sig}\%$ Confidence Interval',
                                 alpha=alpha, edgecolor=color, facecolor=color, linewidth=0)

        plt.title(var_name, {'size': 35}, pad=5.)
        ax.set_xlabel(self.dict[self.date_frequency], {'size': 27}, labelpad=15.)
        ax.set_ylabel('Percentage', {'size': 27}, labelpad=15.)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.xlim(0, hd.shape[1])
        plt.xticks(list(x_ticks)[::20], date_list[::20], rotation=30)
        plt.tick_params(labelsize=25)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Palatino') for label in labels]
        plt.grid(linewidth=1, color='black', alpha=0.1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['top'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

        if save_path is not None:
            plt.savefig(save_path + f'/hd.png', bbox_inches='tight', dpi=200)
        plt.show()
