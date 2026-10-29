"""Statistical tests, multiple-comparison correction and significance annotations.

Pairwise comparisons are returned as "leaves" of the form [(position_1, position_2), p_value].
Leaves can be nested in lists (e.g. one list per stacking level of the significance bars,
and one such list per figure), and `perform_fdr_correction` corrects all leaves of a nested
structure as one family.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05


def _is_leaf(item):
    """Whether `item` is a single comparison of the form [(pos1, pos2), p_value]."""
    return isinstance(item, list) and len(item) == 2 and isinstance(item[0], tuple)


def _iter_leaves(nested):
    """Yield all comparisons in a (possibly nested) list of comparisons."""
    if _is_leaf(nested):
        yield nested
    elif isinstance(nested, list):
        for item in nested:
            yield from _iter_leaves(item)


def compare_two_samples(data1, data2, related=False):
    """Two-sided test of the difference between two samples.

    For related (paired) samples, a paired t-test is used if the paired differences pass a
    Shapiro-Wilk test for normality, otherwise a Wilcoxon signed-rank test. For independent
    samples, an independent t-test is used if both samples pass the Shapiro-Wilk test,
    otherwise a Mann-Whitney U test.

    Parameters
    ----------
    data1, data2 : sequence of float
        The two samples (of equal length and in the same participant order if `related`).
    related : bool, optional
        Whether the samples are paired. Default is False.

    Returns
    -------
    float
        The p-value, or NaN if a sample has fewer than three values.
    """
    data1, data2 = np.asarray(data1, dtype=float), np.asarray(data2, dtype=float)
    if len(data1) < 3 or len(data2) < 3:
        return np.nan

    if related:
        if sts.shapiro(data1 - data2).pvalue > ALPHA:
            return sts.ttest_rel(data1, data2, alternative='two-sided').pvalue
        return sts.wilcoxon(data1, data2, alternative='two-sided').pvalue

    if sts.shapiro(data1).pvalue > ALPHA and sts.shapiro(data2).pvalue > ALPHA:
        return sts.ttest_ind(data1, data2, alternative='two-sided').pvalue
    return sts.mannwhitneyu(data1, data2, alternative='two-sided').pvalue


def test_against_zero(values):
    """Two-sided test of whether the values differ from zero.

    A one-sample t-test is used if the values pass a Shapiro-Wilk test for normality,
    otherwise a Wilcoxon signed-rank test.

    Parameters
    ----------
    values : sequence of float
        The values (e.g. per-participant attention modulation coefficients).

    Returns
    -------
    float
        The p-value, or NaN if there are fewer than three values.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.nan
    if sts.shapiro(values).pvalue > ALPHA:
        return sts.ttest_1samp(values, 0).pvalue
    return sts.wilcoxon(values).pvalue


def determine_significant_combinations(data, positions=None, related=False):
    """Test all pairwise combinations of the samples in `data` with `compare_two_samples`.

    Parameters
    ----------
    data : list of sequences
        One sample per condition.
    positions : list, optional
        Box positions of the conditions in `data`, used to label the comparisons. Default is
        the 1-based index of each condition.
    related : bool, optional
        Whether the samples are paired (same participants in the same order). Default is False.

    Returns
    -------
    list
        One comparison [(position_1, position_2), p_value] per pair of conditions (not
        filtered by significance), ordered from the widest to the narrowest pair.
    """
    if positions is None:
        positions = list(range(1, len(data) + 1))

    pairs = [(i, i + j) for j in reversed(range(1, len(data))) for i in range(len(data) - j)]
    return [[(positions[a], positions[b]), compare_two_samples(data[a], data[b], related=related)] for a, b in pairs]


def perform_fdr_correction(sig_combinations, alpha=ALPHA):
    """Benjamini-Hochberg correction of all p-values in a (possibly nested) list of comparisons.

    All comparisons contained in `sig_combinations` form one family, so pass the comparisons
    of all figures of one metric together to correct across the whole dataset. Comparisons
    with a NaN p-value are left unchanged and do not count towards the family size.

    Parameters
    ----------
    sig_combinations : list
        A comparison [(pos1, pos2), p_value], or an arbitrarily nested list of comparisons.
    alpha : float, optional
        Family-wise false discovery rate. Default is 0.05.

    Returns
    -------
    list
        `sig_combinations`, with the p-values replaced by the corrected ones (in place).
    """
    leaves = [leaf for leaf in _iter_leaves(sig_combinations) if np.isfinite(leaf[1])]
    if leaves:
        _, p_corrected, _, _ = multipletests([leaf[1] for leaf in leaves], alpha=alpha, method='fdr_bh')
        for leaf, p in zip(leaves, p_corrected):
            leaf[1] = p
    return sig_combinations


def p_to_stars(p):
    """Return '***', '**', '*' or 'ns' for p < 0.001, 0.01, 0.05 or larger (and NaN)."""
    if not np.isfinite(p) or p >= 0.05:
        return 'ns'
    return '***' if p < 0.001 else '**' if p < 0.01 else '*'


def plot_significance_bars(bp, sig_combinations, distance=0.5, color='black'):
    """Draw bars with asterisks between significantly different boxes of a boxplot.

    Comparisons with p >= 0.05 are skipped. `sig_combinations` is a list of levels, each a
    list of comparisons; the bars of the first level are drawn highest. Positions in the
    comparisons are 1-based box positions (box k is drawn at x = k - 1).

    Parameters
    ----------
    bp : dict
        Boxplot elements as returned by `matplotlib.pyplot.boxplot`; the whiskers define the
        height above which the bars are drawn.
    sig_combinations : list
        List of levels of comparisons (a flat list of comparisons is treated as one level).
    distance : float, optional
        Vertical spacing between levels as a fraction of the whisker range. Default is 0.5.
    color : str, optional
        Colour of bars and asterisks. Default is 'black'.
    """
    if any(_is_leaf(item) for item in sig_combinations):
        sig_combinations = [sig_combinations]

    whisker_heights = [whisker.get_ydata()[1] for whisker in bp['whiskers']]
    bottom, top = min(whisker_heights), max(whisker_heights)
    y_range = top - bottom

    for l, level_combs in enumerate(sig_combinations):
        significant = [comb for comb in _iter_leaves(level_combs) if np.isfinite(comb[1]) and comb[1] < 0.05]
        if significant:
            plt.autoscale(enable=True, axis='y')

        for comb in significant:
            x1 = comb[0][0] - 1 + 0.02
            x2 = comb[0][1] - 1 - 0.02
            bar_height = (y_range * distance * (len(sig_combinations) - l)) + top
            bar_tips = bar_height - (y_range * 0.05)
            plt.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], c=color, lw=1.5)
            plt.text((x1 + x2) / 2, bar_height + (y_range * 0.002), p_to_stars(comb[1]), ha='center', va='bottom',
                     c=color, fontsize=22)


def remove_participants_with_nones(data):
    """Remove a participant from every condition if its value is None in any condition, so
    that paired samples stay aligned.

    Parameters
    ----------
    data : dict
        {condition: list of per-participant values}, with the same participant order in every
        condition and None for missing values.

    Returns
    -------
    dict
        `data` without the affected participants (modified in place).
    """
    missing = {i for values in data.values() for i, val in enumerate(values) if val is None}
    for cond in data:
        data[cond] = [val for i, val in enumerate(data[cond]) if i not in missing]
    return data
