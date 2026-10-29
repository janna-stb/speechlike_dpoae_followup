"""Box and bar plots of the DPOAE statistics, with significance tests.

`DPOAEplotter` plots the data prepared by `DPOAEStatistics` and annotates the boxplots with
the results of pairwise tests (attended vs. ignored and between voices) or, for the attention
modulation coefficient (AMC), of tests against zero. P-values are Benjamini-Hochberg
corrected across all comparisons of one metric within one call, i.e. across all figures of
that metric, not per figure.

Conditions are grouped into figures according to `sort_by`:

- 'JS26': one figure per harmonic condition (both voices and attention conditions).
- 'JS26_speaker': one figure per harmonic condition and voice.
- 'JS25': one figure per harmonic type (resolved/unresolved) and voice.
- 'MSA21': one figure per voice.
- '<dataset>_wo_attention': one figure with one box per harmonic condition and voice; the
  values of all attention conditions are pooled (concatenated).
- '<dataset>_pooled_attention': as '_wo_attention', but the attention conditions are averaged
  per participant, so that each participant contributes one value per box.
"""

import re
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.ticker import ScalarFormatter

from functions_significance_testing import (determine_significant_combinations, perform_fdr_correction,
                                            plot_significance_bars, remove_participants_with_nones,
                                            test_against_zero, p_to_stars)

# harmonic conditions of each dataset as they appear in the condition names
HARMONIC_CONDITIONS = {'JS25': ['Resolved', 'Unresolved'],
                       'JS26': ['Low_res', 'Low_unres', 'Trans_res', 'Trans_unres']}
VOICES = ['F', 'M']

# abbreviations used for the tick labels
TICK_LABELS = {'Attended': 'Atd', 'Ignored': 'Ign', 'Baseline_DPs': 'BL', 'Resolved': 'res', 'Unresolved': 'unres',
               'Low_res': 'LR', 'Low_unres': 'LU', 'Trans_res': 'TR', 'Trans_unres': 'TU', 'female': 'F', 'male': 'M'}

YLABELS = {'PSD_dB': 'PSD (dB SPL)', 'PSD_Pa': 'PSD (Pa)', 'mag': 'Magnitude', 'snr': 'SNR [dB]', 'delay': 'Delay [ms]',
           'auc': 'AUC', 'width': 'Width', 'sigma': 'Sigma from Fit [ms]', 'fwhm': 'FWHM from Fit [ms]',
           'GAmag': 'Magnitude from Fit', 'MSCmean': 'Mean MSC', 'MSCmax': 'Max MSC', 'MSCpw': 'Power-weighted MSC'}


def _tick_label(cond):
    """Abbreviate a condition name for the x-axis, e.g. 'Low_res F Attended' -> 'LR F Atd'."""
    return re.sub(r'\b(?:' + '|'.join(TICK_LABELS) + r')\b', lambda m: TICK_LABELS[m.group()], cond)


def _pool_conditions(conditions, data, new_conditions, average):
    """Pool the values of all conditions that contain every word of a new condition name.

    Parameters
    ----------
    conditions : list of str
        Original condition names.
    data : dict
        {condition: list of values}.
    new_conditions : list of str
        Names of the pooled conditions, e.g. 'Low_res F'.
    average : bool
        If True, average the matched conditions per participant (position-wise); otherwise
        concatenate them.

    Returns
    -------
    dict
        {new condition: list of values}.
    """
    pooled = {}
    for new_cond in new_conditions:
        keywords = new_cond.split()
        matched = [cond for cond in conditions if all(keyword in cond for keyword in keywords)]
        if average:
            values = [data[cond] for cond in matched]
            if len({len(v) for v in values}) > 1:
                raise ValueError(f"Cannot average {matched} for '{new_cond}': they have different numbers of values, "
                                 f"so index i is not guaranteed to be the same participant in every condition.")
            pooled[new_cond] = list(np.nanmean(np.array(values, dtype=float), axis=0))
        else:
            pooled[new_cond] = []
            for cond in matched:
                if isinstance(data[cond], Iterable) and not isinstance(data[cond], str):
                    pooled[new_cond].extend(data[cond])
                else:
                    pooled[new_cond].append(data[cond])
    return pooled


class DPOAEplotter:
    """Create styled boxplots and barplots from the data prepared by `DPOAEStatistics`.

    Colour encodes harmonic condition and voice, hatching and opacity the attention condition,
    and the marker of the mean the voice. Plots are saved as .png files in subfolders of
    `data_path` (`Boxplots_<subsample>`, `Barplots_<subsample>`).

    Parameters
    ----------
    data_path : str or Path
        Folder the plots are saved to. Created if it does not exist.
    """

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.plot_path = self.data_path
        self.plot_path.mkdir(parents=True, exist_ok=True)

        self.fontsize = 22
        self.markersize = 10
        self.median_linewidth = 3
        self.linewidth = 1.2

        plt.rcParams.update({
            'font.size': 18,
            'axes.titlesize': 20,
            'axes.labelsize': 30,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 18,
            'axes.linewidth': 1.5,
            'xtick.major.size': 10,
            'ytick.major.size': 10,
            'xtick.major.width': 2,
            'ytick.major.width': 2,
        })

        # JS26 colours as in the figures of the paper on the JS26 experiment; JS25 colours as in the
        # Frontiers paper; 'x' is used for MSA21, which has no harmonic conditions
        self.color_palettes = {'low_res': {'F': '#355C8A', 'M': '#4797B8'},
                               'low_unres': {'F': '#26796B', 'M': '#8BA64A'},
                               'trans_res': {'F': '#BD5038', 'M': '#E29A36'},
                               'trans_unres': {'F': '#7653A3', 'M': '#B65D91'},
                               'resolved': {'M': '#a6dba0', 'F': '#c2a5cf'},
                               'unresolved': {'M': '#66c2a5', 'F': '#8da0cb'},
                               'x': {'M': 'blue', 'F': 'orange'}}

        self.attention_style = {'attended': {'alpha': 0.95, 'hatch': None},
                                'ignored': {'alpha': 0.45, 'hatch': '//'},
                                'baseline_dps': {'alpha': 0.45, 'hatch': '..'}}

        self.speaker_style = {'F': {'marker': 'o'}, 'M': {'marker': 's'}}

    def sort_conditions_for_plotting(self, conditions, data=None, sort_by='JS26'):
        """Group condition names into figures and create their tick labels.

        Parameters
        ----------
        conditions : list of str
            Condition names, e.g. 'Low_res F Attended'.
        data : dict, optional
            {condition: values}. Required for the '_wo_attention' and '_pooled_attention'
            groupings, which create new pooled conditions (see the module docstring).
        sort_by : str, optional
            Grouping, see the module docstring. Default is 'JS26'.

        Returns
        -------
        sorted_conditions : list of list of str
            Condition names per figure.
        ax_names : list of list of str
            Tick labels matching `sorted_conditions`.
        sorted_data : dict
            Only returned if `data` is given: {condition: values} for the conditions in
            `sorted_conditions`.
        """
        dataset, _, variant = sort_by.partition('_')
        if dataset not in ('JS25', 'JS26', 'MSA21') or variant not in ('', 'speaker', 'wo_attention', 'pooled_attention') \
                or (variant == 'speaker' and dataset != 'JS26'):
            raise ValueError(f'Invalid sort_by value: {sort_by}.')

        sorted_conditions, ax_names, sorted_data = [], [], data

        if variant in ('wo_attention', 'pooled_attention'):
            if data is None:
                raise ValueError(f"sort_by='{sort_by}' requires data.")
            levels = [''] if dataset == 'MSA21' else HARMONIC_CONDITIONS[dataset]
            new_conditions = sorted(f'{hl} {sp}'.strip() for hl in levels for sp in VOICES)
            sorted_data = _pool_conditions(conditions, data, new_conditions, average=(variant == 'pooled_attention'))
            sorted_conditions = [new_conditions]
            ax_names = [[_tick_label(cond) for cond in new_conditions]]

        elif dataset == 'JS26':
            # condition names: '<harm_cond> <voice> [<attention>]'
            for level in HARMONIC_CONDITIONS['JS26']:
                for speakers in ([[v] for v in VOICES] if variant == 'speaker' else [VOICES]):
                    sub_conds = sorted(c for c in conditions if c.split()[0] == level and c.split()[1] in speakers)
                    sorted_conditions.append(sub_conds)
                    ax_names.append([_tick_label(c) for c in sub_conds])

        elif dataset == 'JS25':
            # condition names: '<Resolved|Unresolved> <voice> [<attention>]'
            for harm in HARMONIC_CONDITIONS['JS25']:
                for g in ['M', 'F']:
                    sub_conds = sorted(c for c in conditions if c.split()[0] == harm and c.split()[1] == g)
                    if len(sub_conds) == 3:  # order Attended, Ignored, Baseline
                        sub_conds[1], sub_conds[2] = sub_conds[2], sub_conds[1]
                    sorted_conditions.append(sub_conds)
                    ax_names.append([_tick_label(' '.join(c.split(' ')[1:])) for c in sub_conds])

        else:  # MSA21, condition names: '<voice> [<attention>]'
            for g in VOICES:
                sub_conds = sorted(c for c in conditions if c.split()[0] == g)
                sorted_conditions.append(sub_conds)
                ax_names.append([_tick_label(c) for c in sub_conds])

        if data is not None:
            return sorted_conditions, ax_names, sorted_data
        return sorted_conditions, ax_names

    def _condition_styles(self, cond_list, sort_by):
        """Colours, hatches, opacities and markers of the boxes/bars of one figure.

        Returns
        -------
        colors, hatches, opacity, markers : list
            One entry per condition.
        harm_type : str
            Harmonic condition of the last condition (used in titles and file names).
        """
        colors, hatches, opacity, markers = [], [], [], []
        harm_type = 'x'
        for name in cond_list:
            if sort_by.startswith('MSA21'):
                name = 'x ' + name
            words = name.split()
            harm_type, voice = words[0].lower(), words[1]
            att = words[2].lower() if len(words) > 2 else 'attended'
            hatches.append(self.attention_style[att]['hatch'])
            opacity.append(self.attention_style[att]['alpha'])
            markers.append(self.speaker_style[voice]['marker'])
            colors.append(self.color_palettes[harm_type][voice])
        return colors, hatches, opacity, markers, harm_type

    def _style_boxplot(self, ax, bp, data_to_plot, colors, hatches, opacity, markers):
        """Apply colours, hatches and markers to a boxplot and draw the means."""
        for patch, color, hatch, alpha in zip(bp['boxes'], colors, hatches, opacity):
            patch.set_facecolor(to_rgba(color, alpha))
            patch.set_edgecolor(to_rgba(color, 1.0))
            patch.set_hatch(hatch)
            patch.set_linewidth(self.linewidth)
        for line, color in zip(bp['whiskers'] + bp['caps'], list(np.repeat(colors, 2)) * 2):
            line.set_color(to_rgba(color, 1.0))
            line.set_linewidth(self.linewidth)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(self.median_linewidth)
        for x, y, color, marker in zip(range(len(data_to_plot)), data_to_plot, colors, markers):
            ax.scatter(x, np.nanmean(y), marker=marker, s=60, facecolor='white', edgecolor=to_rgba(color, 1.0),
                       linewidth=self.linewidth, zorder=3)
        for flier, color, marker in zip(bp['fliers'], colors, markers):
            flier.set(marker=marker, markerfacecolor='white', markeredgecolor=to_rgba(color, 1.0), markeredgewidth=1.2,
                      markersize=6, alpha=0.9, linestyle='none')

    @staticmethod
    def _pairwise_comparisons(data, related):
        """Pairwise tests within one figure, grouped into levels of significance bars.

        - 4 boxes (e.g. F Atd, F Ign, M Atd, M Ign): 1 vs 3 and 2 vs 4 (between voices), and
          1 vs 2 and 3 vs 4 (attended vs ignored).
        - 3 boxes: 1 vs 3, and 1 vs 2 and 2 vs 3.
        - 8 boxes: neighbouring pairs 1 vs 2, 3 vs 4, 5 vs 6, 7 vs 8.
        - otherwise: all pairs.

        Returns
        -------
        list
            List of levels, each a list of comparisons [(pos1, pos2), p_value] (1-based positions).
        """
        def test(pairs):
            out = []
            for a, b in pairs:
                out += determine_significant_combinations([data[a - 1], data[b - 1]], positions=[a, b], related=related)
            return out

        n = len(data)
        if n == 4:
            return [test([(1, 3)]), test([(2, 4)]), test([(1, 2), (3, 4)])]
        if n == 3:
            return [test([(1, 3)]), test([(1, 2), (2, 3)])]
        if n == 8:
            return [test([(1, 2), (3, 4), (5, 6), (7, 8)])]
        return [determine_significant_combinations(data, positions=list(range(1, n + 1)), related=related)]

    def plot_boxplots_general(self, subsample, data_dict, related=True, sort_by='JS26_speaker'):
        """Create one boxplot per metric and figure group with significance bars.

        Box fliers (values beyond 1.5 IQR from the quartiles) are excluded from the tests;
        for related data they are removed for all conditions of the figure, so that the
        participants stay paired. P-values are FDR-corrected across all comparisons of a
        metric (all figures together). The metrics 'auc' and 'width' are always tested as
        unpaired, because the GA-peak-based values leave too few complete pairs.
        Figures whose first condition contains only NaN are skipped.

        Parameters
        ----------
        subsample : str
            Name of the output folder, `Boxplots_<subsample>`.
        data_dict : dict
            {metric: {condition: list of values}} as returned by
            `DPOAEStatistics.prepare_boxplot_data_*`.
        related : bool, optional
            Whether the values of different conditions come from the same participants in the
            same order (one value per participant). Default is True.
        sort_by : str, optional
            Grouping of conditions into figures, see the module docstring. Default is 'JS26_speaker'.

        Notes
        -----
        Saves one 'boxplot_<metric>_<harm_type>_<voice>.png' per metric and figure in
        `data_path/Boxplots_<subsample>/`.
        """
        save_path = self.plot_path / f'Boxplots_{subsample}'
        save_path.mkdir(parents=True, exist_ok=True)

        for metric, data in data_dict.items():
            related_metric = related and metric not in ('auc', 'width')
            sorted_conditions, ax_names, sorted_data = self.sort_conditions_for_plotting(list(data.keys()), data,
                                                                                         sort_by=sort_by)
            figures = []  # (fig, ax, bp, comparisons, file name) per figure of this metric

            for ax_name, cond_list in zip(ax_names, sorted_conditions):
                if not cond_list or all(np.isnan(val) for val in sorted_data[cond_list[0]]):
                    continue

                first = cond_list[0].split()
                voice = first[1] if len(first) > 2 else first[0]
                colors, hatches, opacity, markers, harm_type = self._condition_styles(cond_list, sort_by)

                data_bp_dict = {cond: [None if (val is None or np.isnan(val)) else val for val in sorted_data[cond]]
                                for cond in cond_list}
                if related_metric:
                    data_bp_cut = remove_participants_with_nones(data_bp_dict)
                    data_to_plot = [data_bp_cut[cond] for cond in cond_list]
                else:
                    data_to_plot = [[val for val in data_bp_dict[cond] if val is not None] for cond in cond_list]

                fig, ax = plt.subplots(figsize=(8, 6))
                bp = ax.boxplot(data_to_plot, patch_artist=True, positions=np.arange(len(cond_list)), showfliers=True)
                self._style_boxplot(ax, bp, data_to_plot, colors, hatches, opacity, markers)

                # exclude the fliers from the statistical tests
                outlier_idxs = [[i for i, val in enumerate(box) if val in bp['fliers'][k].get_ydata()]
                                for k, box in enumerate(data_to_plot)]
                if related_metric:
                    outliers = {idx for idxs in outlier_idxs for idx in idxs}
                    data_wo_outliers = [[val for i, val in enumerate(box) if i not in outliers] for box in data_to_plot]
                    N = len(data_wo_outliers[0])
                else:
                    data_wo_outliers = [[val for i, val in enumerate(box) if i not in idxs]
                                        for box, idxs in zip(data_to_plot, outlier_idxs)]
                    N = [len(box) for box in data_wo_outliers]

                ax.yaxis.offsetText.set_size(20)
                if metric == 'mag':
                    formatter = ScalarFormatter(useMathText=True)
                    formatter.set_powerlimits((-3, -3))
                    ax.yaxis.set_major_formatter(formatter)
                    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
                ax.tick_params(axis='y', labelsize=self.fontsize)
                ax.set_xticks(np.arange(len(cond_list)))
                if len(cond_list) > 4:
                    rot = 45
                    ax.set_title(f'N={N}', pad=15)
                else:
                    rot = 0
                    ax.set_title(f'{harm_type} {voice} (N={N})', pad=15)
                ax.set_xticklabels(ax_name, fontsize=self.fontsize, rotation=rot)
                ax.set_ylabel(YLABELS[metric], labelpad=15)

                figures.append((fig, ax, bp, self._pairwise_comparisons(data_wo_outliers, related_metric),
                                f'boxplot_{metric}_{harm_type}_{voice}.png'))

            # one FDR family per metric: all comparisons of all figures
            perform_fdr_correction([comparisons for _, _, _, comparisons, _ in figures])

            for fig, ax, bp, comparisons, file_name in figures:
                plt.figure(fig.number)
                plt.sca(ax)
                plot_significance_bars(bp, comparisons, color='black')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(save_path / file_name, dpi=300)
                plt.close(fig)

    def plot_boxplots_amc(self, subsample, data_dict, sort_by='JS26'):
        """Create boxplots of the attention modulation coefficient (AMC), each box tested
        against zero.

        Box fliers are excluded from the tests. Each box is tested with `test_against_zero`
        (one-sample t-test, or Wilcoxon signed-rank test if the values are not normally
        distributed), and the p-values of all boxes of all figures are FDR-corrected together.

        Parameters
        ----------
        subsample : str
            Name of the output folder, `Boxplots_<subsample>`.
        data_dict : dict
            {condition: list of per-participant AMC values} as returned by
            `DPOAEStatistics.convert_boxplot_data_for_amc`.
        sort_by : str, optional
            Grouping of conditions into figures, see the module docstring. Default is 'JS26'.

        Notes
        -----
        Saves one 'boxplot_amc_<harm_type>.png' per figure in `data_path/Boxplots_<subsample>/`.
        """
        save_path = self.plot_path / f'Boxplots_{subsample}'
        save_path.mkdir(parents=True, exist_ok=True)

        sorted_conditions, ax_names, sorted_data = self.sort_conditions_for_plotting(list(data_dict.keys()), data_dict,
                                                                                     sort_by=sort_by)
        figures = []  # (fig, ax, text height, p-values, file name) per figure

        for ax_name, cond_list in zip(ax_names, sorted_conditions):
            if not cond_list:
                continue
            colors, _, _, markers, harm_type = self._condition_styles(cond_list, sort_by)
            bp_data = [[val for val in sorted_data[cond] if val is not None and not np.isnan(val)] for cond in cond_list]

            fig, ax = plt.subplots(figsize=(8, 6))
            bp = ax.boxplot(bp_data, patch_artist=True, positions=np.arange(len(cond_list)), showfliers=True)
            self._style_boxplot(ax, bp, bp_data, colors, [None] * len(colors), [1.0] * len(colors), markers)

            # exclude the fliers from the tests
            clean_bp_data = [list(np.array(box)[~np.isin(box, bp['fliers'][i].get_ydata())]) for i, box in enumerate(bp_data)]
            N = [len(box) for box in clean_bp_data]

            ax.yaxis.offsetText.set_size(20)
            ax.tick_params(axis='y', labelsize=self.fontsize)
            ax.set_xticks(np.arange(len(cond_list)))
            ax.set_xticklabels(ax_name, fontsize=self.fontsize)
            ax.set_ylabel('AMC', labelpad=15)
            ax.set_title(f'AMC {subsample} - {harm_type} (N={N})', pad=15)

            ymin, ymax = ax.get_ylim()
            ax.set_ylim((ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)))
            text_height = ymin - 0.05 * (ymax - ymin)

            pvals = [[(i + 1, i + 1), test_against_zero(vals)] for i, vals in enumerate(clean_bp_data)]
            figures.append((fig, ax, text_height, pvals, f'boxplot_amc_{harm_type}.png'))

        # one FDR family: all boxes of all figures
        perform_fdr_correction([pvals for _, _, _, pvals, _ in figures])

        for fig, ax, text_height, pvals, file_name in figures:
            for i, (_, p) in enumerate(pvals):
                ax.text(i, text_height, p_to_stars(p), ha='center', va='bottom', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            sns.despine(ax=ax)
            fig.tight_layout()
            fig.savefig(save_path / file_name, dpi=1200)
            plt.close(fig)

    def plot_barplots_sigpeaks(self, subsample, data_dict, sort_by='JS26'):
        """Create barplots of the percentage of trials with a significant peak, per method
        (cross-correlation, MSC, PSD) and figure group.

        Parameters
        ----------
        subsample : str
            Name of the output folder, `Barplots_<subsample>`.
        data_dict : dict
            {"no_sig_peaks": ..., "no_all_peaks": ...} as returned by
            `DPOAEStatistics.prepare_barplot_data_sigpeaks`.
        sort_by : str, optional
            Grouping of conditions into figures, see the module docstring. For the JS26 and
            MSA21 '_wo_attention' groupings, all bars are plotted in one figure with an extra
            gap after every second bar. Default is 'JS26'.

        Notes
        -----
        Saves one 'barplot_sig_peaks_<method>(_<harm_type>_<voice>).png' per method and figure
        in `data_path/Barplots_<subsample>/`.
        """
        save_path = self.plot_path / f'Barplots_{subsample}'
        save_path.mkdir(parents=True, exist_ok=True)

        numbers_sig = data_dict['no_sig_peaks']
        numbers_all = data_dict['no_all_peaks']

        for metric in numbers_sig:
            conditions = list(numbers_sig[metric].keys())
            sorted_conditions, ax_names, sorted_numbers_sig = self.sort_conditions_for_plotting(conditions, numbers_sig[metric], sort_by=sort_by)
            _, _, sorted_numbers_all = self.sort_conditions_for_plotting(conditions, numbers_all[metric], sort_by=sort_by)

            for ax_name, cond_list in zip(ax_names, sorted_conditions):
                colors, hatches, opacity, _, harm_type = self._condition_styles(cond_list, sort_by)
                voice = ('x ' + cond_list[-1] if sort_by.startswith('MSA21') else cond_list[-1]).split()[1]

                data = [np.sum(sorted_numbers_sig[cond]) / np.sum(sorted_numbers_all[cond]) * 100 for cond in cond_list]

                if 'wo_attention' in sort_by and not sort_by.startswith('JS25'):
                    # extra gap after every second bar
                    positions = list(np.cumsum([0] + [2 if i % 2 == 1 else 1 for i in range(len(data) - 1)]))
                    title = metric
                    save_name = f'barplot_sig_peaks_{metric}.png'
                else:
                    positions = np.arange(len(data))
                    title = harm_type
                    save_name = f'barplot_sig_peaks_{metric}_{harm_type}' + (f'_{voice}.png' if 'wo_attention' not in sort_by else '.png')

                fig, ax = plt.subplots(figsize=(7, 5))
                bars = ax.bar(positions, data, color=colors, hatch=hatches, linewidth=self.linewidth, width=0.7)
                for bar, color, alpha in zip(bars, colors, opacity):
                    bar.set_facecolor(to_rgba(color, alpha))
                    bar.set_edgecolor(to_rgba(color, 1.0))

                ax.set_ylim(0, 100)
                ax.set_ylabel('Percentage', labelpad=15)
                ax.tick_params(labelsize=self.fontsize)
                ax.set_xticks(positions)
                ax.set_xticklabels(ax_name, fontsize=14, rotation=45)
                ax.set_title(title, pad=15, fontsize=self.fontsize)

                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(save_path / save_name, dpi=1200)
                plt.close(fig)
