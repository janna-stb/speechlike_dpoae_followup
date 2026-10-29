"""Grand averages and boxplot/barplot data for the statistical evaluation of a dataset.

`DPOAEStatistics` reads the per-condition file lists written by `sort_files_to_conditions.py`
and computes grand averages, Gaussian fits of the grand-average cross-correlations, and
per-participant or per-trial values of all metrics, which are plotted and tested by
`DPOAEplotter`. See the README for the order in which the methods are called.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from plot_grand_averages import plot_GA_crosscorr, plot_GA_msc, plot_GA_psd, find_peak_amplitude, determine_noise_level

# np.trapz was renamed to np.trapezoid in NumPy 2.0 (and later removed)
trapezoid = getattr(np, 'trapezoid', None) or np.trapz

# harmonic conditions of each dataset as they appear in the condition names
HARMONIC_CONDITIONS = {'JS25': ['Resolved', 'Unresolved'],
                       'JS26': ['Low_res', 'Trans_res', 'Low_unres', 'Trans_unres'],
                       'MSA21': ['']}


def participant_from_path(file):
    """Return the participant ID from the `results_<participant>` folder in a result file's path."""
    for part in reversed(Path(file).parts):
        if part.startswith('results_'):
            return part[len('results_'):]
    raise ValueError(f'No results_<participant> folder in {file}')

class DPOAEStatistics:
    """Compute grand averages and prepare statistics/plot data for DPOAE cross-correlation,
    magnitude-squared coherence (MSC), and power spectral density (PSD) analyses.

    Results (grand averages, fits, and boxplot/barplot data) are written to disk as
    .npy and .csv files under `res_path`, organized into subfolders per method/analysis.

    The averaged autocorrelations of the DP waveforms (written by
    `DPOAEProcessor.compute_autocorrelations`) are expected in
    `<dataset folder>/results_Autocorrelations/GA_autocorr_dict.npz`, where the dataset
    folder is the parent of the 'Statistical_Analysis' folder that contains
    `files_per_condition_path`.
    """

    def __init__(self, files_per_condition_path, res_path, participants, ga_reference_path=None):
        """
        Parameters
        ----------
        files_per_condition_path : str or Path
            Path to a .npy file containing a dict of the form
            {condition: {data_key: [file paths, ...], ...}, ...}, mapping each condition
            to the files holding its per-trial crosscorr/msc/psd results.
        res_path : str or Path
            Directory where computed results (grand averages, statistics, plots) are saved.
            Created if it does not already exist.
        participants : list of str
            Participant IDs (e.g. 'P01' or 'outofear_01') used to group files by participant.
        ga_reference_path : str or Path, optional
            Only needed for out-of-ear (control) recordings. Statistical_Analysis folder of
            the corresponding in-ear dataset (e.g. '<data>/JS26/Statistical_Analysis/P01_to_P44').
            Out-of-ear data contain no DPOAE peak, so the peak position is taken from the
            in-ear grand average stored under `ga_reference_path / res_path.name`.
        """
        self.files_per_condition_path = Path(files_per_condition_path)
        self.files_per_condition = np.load(self.files_per_condition_path, allow_pickle=True).item()
        self.conditions = self.files_per_condition.keys()
        self.res_path = Path(res_path)
        self.participants = participants
        self.ga_reference_path = Path(ga_reference_path) if ga_reference_path is not None else None

        self.fs = 44100  # Sampling frequency
        self.sig_border = 97 # percentile threshold for significance

        if not self.res_path.exists():
            self.res_path.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update({
            'font.size': 14,          # General font size
            'axes.titlesize': 16,     # Title font size
            'axes.labelsize': 16,     # X and Y label font size
            'xtick.labelsize': 12,    # X-tick label size
            'ytick.labelsize': 14,    # Y-tick label size
            'legend.fontsize': 14     # Legend font size
        })

        parts = self.files_per_condition_path.parts
        dataset_root = Path(*parts[:parts.index('Statistical_Analysis')])
        self.autocorr_path = dataset_root / 'results_Autocorrelations' / 'GA_autocorr_dict.npz'

    def compute_grand_average(self, method='crosscorr', per_participant=False, fit_GA=False, distinguish_attention=True):
        """Compute the grand average (GA) of the specified method across trials, save the GA
        data and summary statistics to disk, and plot each condition's grand average.

        For method='crosscorr', a Gaussian may additionally be fit to each condition's GA
        envelope (see `fit_GA`).

        Parameters
        ----------
        method : str, default 'crosscorr'
            The method to compute the grand average for. One of 'crosscorr', 'msc',
            'msc_corrected', or 'psd'.
        per_participant : bool, default False
            If True, average each participant's own per-participant grand average first
            (files under the 'GA_<method>' key), so every participant contributes equally
            to the final grand average. If False, average directly across all trial files.
        fit_GA : bool, default False
            If True, fit a Gaussian to the grand average envelope for each condition and
            save the fit statistics. Only implemented for method='crosscorr'; raises
            ValueError otherwise.
        distinguish_attention : bool, default True
            If True, compute one grand average per condition as-is. If False, pool the two
            attention conditions (Attended/Ignored) belonging to the same base condition
            before averaging.

        Returns
        -------
        None
            Results are written to `res_path`: the grand average arrays (`GA_<method>_data.npy`),
            summary statistics (`GA_<method>_stats.csv`), per-condition plots, and, if
            `fit_GA` is True, the Gaussian fit results (`GA_fit_<method>_data.npy`,
            `GA_fit_<method>_stats.csv`).
        """

        if method != 'crosscorr' and fit_GA:
            raise ValueError("Gaussian fitting is only implemented for the 'crosscorr' method.")
        
        GA_per_condition = {cond: [] for cond in self.conditions}
        GA_stats = {cond: [] for cond in self.conditions}
        fit_stats_pP = {p: {cond: [] for cond in self.conditions} for p in self.participants} if per_participant and fit_GA else None

        fit_stats_pP_columns = ['Participant', 'Condition', 'GA Fit Amplitude', 'GA Fit Center [frames]', 'GA Fit Center [ms]', 'GA Fit Sigma [frames]', 'GA Fit Sigma [ms]', 'Sigma Autocorrelation [frames]', 'GA Fit Baseline', 'GA Fit R²', 'GA Fit FWHM', 'Converged']

        GA_autocorr_dict = np.load(self.autocorr_path, allow_pickle=True)
        GA_autocorr_new = {}
        for cond in self.conditions:
            autocorr_cond = cond.replace('Baseline_DPs', 'Attended') if 'Baseline_DPs' in cond else cond
            GA_autocorr_new[cond] = GA_autocorr_dict[autocorr_cond]

        folder_name = f'GA_{method}_pP' if per_participant else f'GA_{method}'
        folder_name += '' if distinguish_attention else '_pooled_attention'
        GA_save_path = self.res_path / folder_name
        if not GA_save_path.exists():
            GA_save_path.mkdir(parents=True, exist_ok=True)

        for cond in self.conditions:
            data_per_cond = []
            if method == 'msc_corrected':
                key = 'GA_msc' if per_participant else 'msc'
            else:
                key = 'GA_' + method if per_participant else method

            if not distinguish_attention:
                if len(cond.split()) < 3:
                    base_cond = cond.split()[0]
                else:
                    base_cond = ' '.join(cond.split()[0:2])
                # extract conditions that contain base_cond
                conds_to_average = [c for c in self.conditions if base_cond in c]
                all_files = [file for c in conds_to_average for file in self.files_per_condition[c][key]]
            else:
                all_files = self.files_per_condition[cond][key]

            for file in all_files:
                if method == 'msc' or method == 'msc_corrected':
                    data_dict = np.load(file, allow_pickle=True).item()
                    data = data_dict[f'{method}'] # or msc_corrected as key
                elif method == 'psd':
                    unit = 'dB' # or 'Pa2_per_Hz'
                    data_dict = np.load(file, allow_pickle=True).item()
                    data = data_dict[f'psd_{unit}']
                elif method == 'crosscorr':
                    data = np.load(file, allow_pickle=True)
                    if per_participant and fit_GA:
                        # perform Gaussian fit to individual participant's GA data and extract sigma from fit results
                        lag_ms = data[0, :] / self.fs * 1000
                        lag_peak_ms, mag_peak = find_peak_amplitude(data[3], lag_ms)
                        expected_peak = np.where(lag_ms == lag_peak_ms)[0][0]
                        zero_lag_sample = np.argmin(np.abs(data[0]))
                        fit_stats, fit_vals = self.fit_raw_GAs(data[3], GA_autocorr_new[cond][3], expected_peak_frame=expected_peak, zero_lag_sample=zero_lag_sample, plot_path=None, time_axis=data[0]/self.fs)
                        p = participant_from_path(file)
                        res_pP_df = pd.DataFrame([[p, cond] + list(fit_stats)], columns=fit_stats_pP_columns)
                        res_pP_df.to_csv(file.with_name('GA_fit_stats.csv'), index=False, sep=';')
                        fit_stats_pP[p][cond] = fit_stats
                else:
                    raise ValueError(f"Unknown method: {method}. Choose from 'crosscorr', 'msc', 'msc_corrected' or 'psd'.")
                data_per_cond.append(data)

            GA_per_condition[cond] = np.mean(np.stack(data_per_cond, axis=0), axis=0)
            if method == 'msc' or method == 'msc_corrected':
                plot_GA_msc(cond, data_dict['freqs'], GA_per_condition[cond], GA_save_path / f"GA_{method}_{'_'.join(cond.split())}.png", band_mask=data_dict['band_mask'])
            elif method == 'psd':
                plot_GA_psd(cond, data_dict['freqs'], GA_per_condition[cond], GA_save_path / f"GA_{method}_{'_'.join(cond.split())}.png", unit=unit)
            elif method == 'crosscorr':
                GA_stats[cond] = plot_GA_crosscorr(cond, GA_per_condition[cond], GA_save_path / f"GA_{method}_{'_'.join(cond.split())}.png")
                GA_stats[cond].append(len(data_per_cond))
            else:
                raise ValueError(f"Unknown method: {method}. Choose from 'crosscorr', 'msc', 'msc_corrected' or 'psd'.")

        np.save(GA_save_path / f'GA_{method}_data.npy', GA_per_condition, allow_pickle=True)

        if method == 'crosscorr':
            GA_columns = ['Condition', 'ENVELOPE Lag Peak [ms]', 'ENVELOPE Magnitude Peak', 'ENVELOPE Noise Floor', 'ENVELOPE SNR [dB]', 'ENVELOPE Peak z [SD]', 'Count']
            r, c = np.array([*GA_stats.values()]).shape
            GA_stats_array = np.concatenate((np.array(list(self.conditions)).reshape(r, 1), np.array([*GA_stats.values()])), axis=1)

            # Save the grand average statistics to a csv file
            GA_stats_df = pd.DataFrame(GA_stats_array, columns=GA_columns)
            GA_stats_df.to_csv(GA_save_path / f'GA_{method}_stats.csv', index=False, sep=';')

            if per_participant and fit_GA:
                fit_stats_pP_array = []
                for p, cond_dict in fit_stats_pP.items():
                    for cond, stats in cond_dict.items():
                        fit_stats_pP_array.append([p, cond] + list(stats))
                fit_stats_pP_df = pd.DataFrame(fit_stats_pP_array, columns=fit_stats_pP_columns)
                fit_stats_pP_df.to_csv(GA_save_path / f'GA_fit_{method}_stats_per_participant.csv', index=False, sep=';')

        if fit_GA:
            GA_stats_fit = {cond: [] for cond in self.conditions}
            GA_fits = {cond: [] for cond in self.conditions}

            for cond, means in GA_per_condition.items():
                peak_pos_frames = np.round(GA_stats[cond][0] / 1000 * self.fs)
                expected_peak = np.where(np.round(means[0]) == peak_pos_frames)[0][0]
                zero_lag_sample = np.argmin(np.abs(means[0]))
                if means[0][zero_lag_sample] != 0:
                    print(f"Warning: Zero lag sample for condition {cond} is not exactly zero. Found value: {means[0][zero_lag_sample]}. Check the lag axis of the means for this condition.")
                fit_stats, fit_vals = self.fit_raw_GAs(means[3], GA_autocorr_new[cond][3], expected_peak_frame=expected_peak, zero_lag_sample=zero_lag_sample, plot_path=GA_save_path, time_axis=means[0]/self.fs, plot_title=f"{method} {cond}")
                GA_stats_fit[cond] = fit_stats
                GA_fits[cond] = fit_vals

            np.save(GA_save_path / f'GA_fit_{method}_data.npy', GA_fits, allow_pickle=True)

            GA_fit_columns = ['Condition', 'GA Fit Amplitude', 'GA Fit Center [frames]', 'GA Fit Center [ms]', 'GA Fit Sigma [frames]', 'GA Fit Sigma [ms]', 'Sigma Autocorrelation [frames]', 'GA Fit Baseline', 'GA Fit R²', 'GA Fit FWHM', 'Converged']             
            rfit, cfit = np.array([*GA_stats_fit.values()]).shape
            GA_stats_fit_array = np.concatenate((np.array(list(self.conditions)).reshape(rfit, 1), np.array([*GA_stats_fit.values()])), axis=1)

            # Save the grand average fit statistics to a CSV file
            GA_stats_fit_df = pd.DataFrame(GA_stats_fit_array, columns=GA_fit_columns)
            GA_stats_fit_df.to_csv(GA_save_path / f'GA_fit_{method}_stats.csv', index=False, sep=';')
        return

    def compute_auc(self, signal, ga_peak_pos, half_width_frames, noise_floor=None):
        """Compute the area under the curve (AUC) and width of the envelope of the peak.
        
        Parameters
        ----------
        signal : numpy.ndarray
            The DPOAE cross-correlation in the form [lag, real, imag, envelope].
        ga_peak_pos : int
            The position of the GA peak.
        half_width_frames : int
            The half-width of the window in frames.
        noise_floor : float, optional
            The noise floor for the signal.

        Returns
        -------
        dict
            A dictionary containing the AUC, width, amplitude, and truncation information.
        """
        n = signal.shape[1]
        env = signal[3, :]
        lo, hi = self._window_indices(n, ga_peak_pos, half_width_frames)

        truncated = (ga_peak_pos - half_width_frames < 0) or (ga_peak_pos + half_width_frames > n-1)
        if truncated:
            print("Warning: Window around GA peak is truncated. Consider using a smaller half_width_frames or excluding this file from AUC analysis.")

        env_win = env[lo:hi+1]

        if noise_floor is None:
            env_corr = env_win
            env_amp = env[ga_peak_pos]
        else:
            env_corr = env_win - noise_floor
            env_amp = env[ga_peak_pos] - noise_floor

        env_auc = trapezoid(env_corr)

        if env_auc < 0:
            env_auc = np.nan

        env_width = env_auc / (env_amp * np.sqrt(2 * np.pi)) if env_auc > 0 else np.nan

        return {
            "auc": env_auc,
            "width": env_width,
            "amp": env_amp,
            "truncated": truncated
        }

    def fit_raw_GAs(self, raw_GA_data, auto_corr_data, expected_peak_frame=None, zero_lag_sample=None, fit_window_sigma_mult=4.0, plot_path=None, time_axis=None, plot_title=None):
        """Fit the raw grand averages with parameters extracted from the autocorrelations of the DP waveforms.
        
        Parameters
        ----------
        raw_GA_data : np.ndarray
            The raw grand average envelope to fit.
        auto_corr_data : np.ndarray
            The autocorrelation data.
        expected_peak_frame : int, optional
            The expected frame of the peak.
        zero_lag_sample : int, optional
            The zero-lag sample.
        fit_window_sigma_mult : float, optional
            The multiplier for the fit window size.
        plot_path : str, optional
            The path to save the plot.
        time_axis : np.ndarray, optional
            The time axis for the plot.
        plot_title : str, optional
            The title for the plot. Also used to derive the saved plot's filename.

        Returns
        -------
        fit_results : tuple
            (amplitude, center, center_display, sigma, sigma_ms, sigma_init, baseline,
            r_squared, fwhm, converged). `center` is in frames, `center_display` is in the
            units of `time_axis` if given (ms) else frames. `sigma_init` is the seed value
            estimated from `auto_corr_data`. `converged` is False if curve_fit failed to
            converge, in which case the initial guesses are returned instead of a fit.
        fits : np.ndarray
            Array of shape (3, n_window_samples) stacking (x_win, y_win, y_fit_win): the
            frame indices, raw data, and fitted curve within the fit window.
        """
        # derive initial sigma from autocorrelation
        sigma_init = self._estimate_sigma_from_autocorr(auto_corr_data)

        # locate peak
        if expected_peak_frame is None:
            peak_idx = int(np.argmax(raw_GA_data))
        else:
            # Search within ± 2*sigma_init of the hint to avoid being pulled by spurious noise peaks far away
            search_half = int(2 * sigma_init)
            lo = max(0, expected_peak_frame - search_half)
            hi = min(len(raw_GA_data), expected_peak_frame + search_half)
            peak_idx = lo + int(np.argmax(raw_GA_data[lo:hi]))

        # define fit window
        half_win = int(np.ceil(fit_window_sigma_mult * sigma_init))
        win_start = max(0, peak_idx - half_win)
        win_stop = min(len(raw_GA_data), peak_idx + half_win)

        x_win = np.arange(win_start, win_stop, dtype=float)
        y_win = raw_GA_data[win_start:win_stop]

        # initial parameter guesses
        amp_init = raw_GA_data[peak_idx]
        baseline_init = np.percentile(y_win, 10)  # robust baseline estimate
        center_init = float(peak_idx)

        p0 = [amp_init, center_init, sigma_init, baseline_init]

        causal_buffer = int(0.001 * self.fs)  
        if zero_lag_sample is None:
            zero_lag_sample = len(raw_GA_data) // 2  - 1 # default to center of the data if not provided

        center_lower = zero_lag_sample - causal_buffer
        if p0[1] < center_lower:
            p0[1] = zero_lag_sample
        
        # Bounds: amplitude and sigma positive; center within window; baseline free
        lower = [0,         win_start, 0.5,       -np.inf]
        upper = [np.inf,     win_stop,   half_win,    np.inf]

        # fit
        converged = True
        try:
            popt, pcov = curve_fit(
                self._gaussian_baseline,
                x_win,
                y_win,
                p0=p0,
                bounds=(lower, upper),
                maxfev=10000,
            )
        except (RuntimeError, ValueError):
            # curve_fit did not converge — return initial guess so analysis can continue; converged flag signals the failure
            popt = np.array(p0)
            converged = False

        amplitude, center, sigma, baseline = popt
        center = int(round(center))  # round center to nearest sample

        # Goodness-of-fit methods                                          
        y_fit_win = self._gaussian_baseline(x_win, *popt)
        residuals = y_win - y_fit_win

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_win - np.mean(y_win)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Derived quantities                                                
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        sigma_ms = sigma / self.fs * 1000.0
        fwhm_ms = fwhm / self.fs * 1000.0

        if time_axis is not None and 0 <= center < len(time_axis):
            center_display = time_axis[center] * 1000
        else:
            center_display = center / self.fs * 1000
        
        # Plot                                                              
        if plot_path is not None:
            if time_axis is not None:
                x_plot_full = time_axis * 1000  # convert to ms for display
                x_plot_win = time_axis[win_start:win_stop] * 1000
                x_label = "Time (ms)"
                sigma_display = sigma_ms
            else:
                x_plot_full = np.arange(len(raw_GA_data))
                x_plot_win = x_win
                x_label = "Lag (samples)"
                center_display = center
                sigma_display = sigma

            fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
            fig.suptitle(plot_title, fontsize=13, fontweight="bold")

            # --- Top panel: full cross-correlation with fit window marked --- #
            ax0 = axes[0]
            ax0.plot(x_plot_full, raw_GA_data, color="#4C72B0", lw=1.2,
                    label="GA cross-correlation envelope")
            ax0.axvspan(x_plot_win[0], x_plot_win[-1], alpha=0.15,
                        color="#DD8452", label="Fit window")
            ax0.set_xlabel(x_label)
            ax0.set_ylabel("Amplitude")
            ax0.set_title("Full GA cross-correlation")
            ax0.legend(fontsize=9)
            ax0.grid(True, alpha=0.3)

            # --- Bottom panel: fit window zoomed in --- #
            ax1 = axes[1]
            ax1.plot(x_plot_win, y_win, "o", color="#4C72B0", ms=3,
                    label="Data (fit window)", zorder=3)
            ax1.plot(x_plot_win, y_fit_win, color="#C44E52", lw=2,
                    label=f"Gaussian fit  (R²={r_squared:.4f})", zorder=4)
            ax1.axvline(center_display, color="#C44E52", lw=1, ls="--", alpha=0.7,
                        label=f"Center = {center_display:.2f}")
            ax1.axhline(amplitude / 2 + baseline, color="#8172B2", lw=1, ls=":",
                        alpha=0.8, label=f"FWHM = {fwhm_ms:.2f} ms")
            ax1.axvspan(
                center_display - sigma_display,
                center_display + sigma_display,
                alpha=0.12, color="#8172B2", label=f"±σ = {sigma_ms:.2f} ms"
            )

            ax1.set_xlabel(x_label)
            ax1.set_ylabel("Amplitude")
            ax1.set_title(
                f"Gaussian fit — amplitude={amplitude:.4f}, "
                f"σ={sigma_ms:.2f} ms, FWHM={fwhm_ms:.2f} ms"
            )
            ax1.legend(fontsize=8, loc="upper left")
            ax1.grid(True, alpha=0.3)

            if not converged:
                fig.text(0.5, 0.01, "WARNING: curve_fit did not converge — "
                        "initial guesses shown", ha="center", color="red", fontsize=10)

            plot_name = f"GA_fit_{'_'.join(plot_title.split())}.png"
            plt.savefig(plot_path / plot_name)
            plt.close()

        fit_results = (amplitude, center, center_display, sigma, sigma_ms, sigma_init, baseline, r_squared, fwhm, converged)
        fits = np.vstack((x_win, y_win, y_fit_win))

        return fit_results, fits

    def _gaussian_baseline(self, x, amplitude, center, sigma, baseline):
        """Gaussian with a constant baseline: A * exp(-(x-c)^2 / (2*sigma^2)) + b.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the function (frame indices).
        amplitude, center, sigma, baseline : float
            Gaussian peak height, peak position, standard deviation, and vertical offset.

        Returns
        -------
        np.ndarray
            The function evaluated at `x`, same shape as `x`.
        """
        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + baseline

    def _estimate_sigma_from_autocorr(self, auto_corr_data):
        """
        Estimate sigma (in samples) from the central peak of the autocorrelation envelope.

        The autocorrelation of the DPOAE waveform peaks at zero lag by definition.
        Its width reflects the temporal spread of the waveform and is the natural
        prior for how wide the cross-correlation peak should be.

        Strategy: find the half-maximum points around the central peak and convert
        the half-width to a Gaussian sigma via sigma = HWHM / sqrt(2 * ln2).

        Parameters
        ----------
        auto_corr_data : np.ndarray
            Full envelope of the autocorrelation of the expected DPOAE waveform. Assumed to
            be a full (two-sided) autocorrelation with the zero-lag peak at its midpoint.

        Returns
        -------
        sigma : float
            Estimated sigma in samples, using `self.fs` for the fallback default. Falls
            back to 5 ms worth of samples if the half-maximum crossing yields an
            implausibly narrow or wide estimate.
        """
        n = len(auto_corr_data)
        center_idx = n // 2  # zero-lag is always at the midpoint for a full autocorrelation

        peak_val = auto_corr_data[center_idx]
        half_max = peak_val / 2.0

        # Walk outward from center to find half-maximum crossing on each side
        left_idx = center_idx
        while left_idx > 0 and auto_corr_data[left_idx] > half_max:
            left_idx -= 1

        right_idx = center_idx
        while right_idx < n - 1 and auto_corr_data[right_idx] > half_max:
            right_idx += 1

        fwhm_samples = right_idx - left_idx
        # Convert FWHM to sigma for a Gaussian: FWHM = 2 * sqrt(2 * ln2) * sigma
        sigma = fwhm_samples / (2 * np.sqrt(2 * np.log(2)))

        # Fallback: if the autocorr peak is too narrow or too wide to estimate reliably, default to 5 ms worth of samples
        if sigma < 1 or sigma > n / 4:
            sigma = 0.005 * self.fs

        return sigma

    def _window_indices(self, n_frames, center_idx, half_width_frames):
        """Compute a symmetric window of indices around `center_idx`, clipped to valid bounds.

        Parameters
        ----------
        n_frames : int
            Total number of frames available (used as the upper bound).
        center_idx : int or float
            Index the window is centered on.
        half_width_frames : int or float
            Half-width of the window in frames.

        Returns
        -------
        lo, hi : int
            Start and (inclusive) end indices of the window, clipped to [0, n_frames - 1].
        """
        lo = max(int(np.floor(center_idx - half_width_frames)), 0)
        hi = min(int(np.ceil(center_idx + half_width_frames)), n_frames - 1)
        return lo, hi

    def prepare_boxplot_data_crosscorr(self, subsample, only_sig=False, ga_peak=False, per_participant=False, fit_peaks=False, avg_participant=True, peak_roi=25, sig_border=None):
        """Extract per-trial cross-correlation peak metrics (SNR, magnitude, delay, and,
        if fitted, AUC/width) per condition, optionally reduced to one value per
        participant, and save/return the result as boxplot-ready data.

        For each trial file, the envelope peak is located (either via `find_peak_amplitude`
        or forced to the grand-average peak position via `ga_peak`), its SNR relative to the
        noise floor is computed, and, if `fit_peaks` is True, the AUC and width around the
        peak are computed via `compute_auc` using the fitted Gaussian sigma from the GA fit
        statistics.

        Parameters
        ----------
        subsample : str
            Name of the grand-average results subfolder to read GA statistics from, e.g.
            'GA_crosscorr', 'GA_crosscorr_pooled_attention', 'GA_crosscorr_pP',
            'GA_crosscorr_pP_pooled_attention'.
        only_sig : bool, default False
            If True, only include trials whose peak magnitude exceeds the noise
            distribution's `self.sig_border` percentile, and save the corresponding file
            paths per condition to a '_significant_trials(_pP).npy' file. Cannot be
            combined with `ga_peak` or `fit_peaks`.
        ga_peak : bool, default False
            If True, evaluate each trial's envelope at the condition's grand-average peak
            lag (from the GA stats CSV) instead of that trial's own peak. Cannot be
            combined with `only_sig`.
        per_participant : bool, default False
            If True, read each trial's per-participant grand-average files (the 'GA_crosscorr'
            key) instead of single-trial files.
        fit_peaks : bool, default False
            If True, use the Gaussian-fit GA statistics (center/sigma) to also compute AUC
            and width per trial via `compute_auc`. Cannot be combined with `only_sig`.
        avg_participant : bool, default True
            If True, average all of a participant's trials per condition first, so each
            participant contributes one value per condition (removes intra-subject
            variability). If False, keep one value per trial.
        peak_roi : float, default 25
            Half-width in ms of the lag window searched for each trial's own peak (used
            unless `ga_peak` is True).
        sig_border : float, optional
            Percentile of the noise region a peak has to exceed for `only_sig`. Default is
            `self.sig_border`.

        Returns
        -------
        dict
            {"snr", "mag", "delay", "auc", "sigma", "fwhm", "GAmag"}, each value a
            dict mapping condition -> list of values (one per participant if
            `avg_participant`, else one per trial). "auc" is only filled if `fit_peaks`
            is True, "sigma"/"fwhm"/"GAmag" (from the per-participant Gaussian fits) only if
            `fit_peaks` and `per_participant` are True, "delay" only if `only_sig` or both
            `fit_peaks` and `per_participant` are True. Also saved to
            `res_path/Boxplot_data_<subsample>/boxplot_data_crosscorr...npy`.
        """

        if sig_border is None:
            sig_border = self.sig_border

        if only_sig and ga_peak:
            raise ValueError("Cannot set both only_sig and ga_peak to True. Choose one or neither.")

        if only_sig and fit_peaks:
            raise ValueError("Cannot set both only_sig and fit_peaks to True (this will result in very few datapoints, because" \
            "likely, only very few individual peaks will match the GA peak position exactly). Choose one or neither.")

        prefix = 'GA_' if per_participant else ''
        if self.participants[0].startswith('outofear'):
            # out-of-ear recordings: use the GA peak position of the corresponding in-ear data
            if self.ga_reference_path is None:
                raise ValueError("Out-of-ear participants detected: pass `ga_reference_path` (the in-ear "
                                 "Statistical_Analysis folder) when creating DPOAEStatistics.")
            ga_path = self.ga_reference_path / self.res_path.name
        else:
            ga_path = self.res_path

        # load info about grand average crosscorrelations from respective csv file
        if fit_peaks:
            ga_info = pd.read_csv(ga_path / subsample / 'GA_fit_crosscorr_stats.csv', sep=';')
            delay_ga = ga_info['GA Fit Center [ms]']
            sigma_autocorr = ga_info.set_index('Condition')['Sigma Autocorrelation [frames]']
        else:
            ga_info = pd.read_csv(ga_path / subsample / 'GA_crosscorr_stats.csv', sep=';')
            delay_ga = ga_info['ENVELOPE Lag Peak [ms]']
            #sigma_ga = np.full_like(delay_ga, fill_value=np.nan)  # Placeholder if not fitting

        significant_files_per_condition = {cond: [] for cond in self.conditions}

        peak_snrs = {cond: [] for cond in self.conditions}
        peak_mags = {cond: [] for cond in self.conditions}
        peak_delays = {cond: [] for cond in self.conditions}
        peak_aucs = {cond: [] for cond in self.conditions}
        peak_sigma = {cond: [] for cond in self.conditions}
        peak_fwhm = {cond: [] for cond in self.conditions}
        peak_GA_mag = {cond: [] for cond in self.conditions}


        for i, key in enumerate(self.conditions):
            all_files = self.files_per_condition[key][f'{prefix}crosscorr']

            px_peak_snrs = {p: [] for p in self.participants}
            px_peak_mags = {p: [] for p in self.participants}
            px_peak_delays = {p: [] for p in self.participants}
            px_peak_aucs = {p: [] for p in self.participants}
            px_peak_sigma = {p: [] for p in self.participants}
            px_peak_fwhm = {p: [] for p in self.participants}
            px_peak_GA_mag = {p: [] for p in self.participants}

            for file in all_files:
                data = np.load(file)
                lag = data[0, :]
                lag_ms = lag / self.fs * 1000
                env = data[3, :]
                if ga_peak:
                    lag_peak_frames = int(delay_ga[i] / 1000 * self.fs) # in frames
                    lag_peak_ms = delay_ga[i]
                    peak_pos = np.where(lag == lag_peak_frames)[0][0]
                    mag_peak = env[peak_pos]
                else:
                    lag_peak_ms, mag_peak = find_peak_amplitude(env, lag_ms, border=peak_roi)
                    lag_peak_frames = int(lag_peak_ms / 1000 * self.fs) # in frames
                    peak_pos = np.where(lag == lag_peak_frames)[0][0]

                sigma_pP = pd.read_csv(file.with_name('GA_fit_stats.csv'), sep=';').set_index('Condition').loc[key, 'GA Fit Sigma [ms]'] if fit_peaks and per_participant else None
                fwhm_pP = pd.read_csv(file.with_name('GA_fit_stats.csv'), sep=';').set_index('Condition').loc[key, 'GA Fit FWHM'] if fit_peaks and per_participant else None
                GA_mag_pP = pd.read_csv(file.with_name('GA_fit_stats.csv'), sep=';').set_index('Condition').loc[key, 'GA Fit Amplitude'] if fit_peaks and per_participant else None

                if per_participant and fit_peaks:
                    lag_peak_ms = pd.read_csv(file.with_name('GA_fit_stats.csv'), sep=';').set_index('Condition').loc[key, 'GA Fit Center [ms]']
                    lag_peak_frames = pd.read_csv(file.with_name('GA_fit_stats.csv'), sep=';').set_index('Condition').loc[key, 'GA Fit Center [frames]']
                    mag_peak = env[peak_pos]

                _, env_noise = determine_noise_level(data)
                noise_floor = np.real(np.mean(env_noise))
                snr = 20 * np.log10(mag_peak / noise_floor)
                sigma = int(np.round(sigma_autocorr[key])) if fit_peaks else None
                auc_results = self.compute_auc(data, peak_pos, sigma, noise_floor=noise_floor) if fit_peaks else []

                if only_sig:
                    if mag_peak <= np.percentile(env_noise, sig_border):
                        continue  # skip trials without a significant peak
                    significant_files_per_condition[key].append(file)

                p = participant_from_path(file)

                px_peak_snrs[p].append(snr)
                px_peak_mags[p].append(mag_peak)
                px_peak_delays[p].append(lag_peak_ms if (per_participant and fit_peaks) or only_sig else np.nan)
                px_peak_aucs[p].append(auc_results['auc'] if auc_results else np.nan)
                px_peak_sigma[p].append(sigma_pP if sigma_pP is not None else np.nan)
                px_peak_fwhm[p].append(fwhm_pP if fwhm_pP is not None else np.nan)
                px_peak_GA_mag[p].append(GA_mag_pP if GA_mag_pP is not None else np.nan)

            if avg_participant:
                peak_snrs[key] = [np.nanmean(px_peak_snrs[p]) for p in self.participants if px_peak_snrs[p]]
                peak_mags[key] = [np.nanmean(px_peak_mags[p]) for p in self.participants if px_peak_mags[p]]
                peak_delays[key] = [np.nanmean(px_peak_delays[p]) for p in self.participants if px_peak_delays[p]]
                if fit_peaks:
                    peak_aucs[key] = [np.nanmean(px_peak_aucs[p]) for p in self.participants if px_peak_aucs[p]]
                    peak_sigma[key] = [np.nanmean(px_peak_sigma[p]) for p in self.participants if px_peak_sigma[p]]
                    peak_fwhm[key] = [np.nanmean(px_peak_fwhm[p]) for p in self.participants if px_peak_fwhm[p]]
                    peak_GA_mag[key] = [np.nanmean(px_peak_GA_mag[p]) for p in self.participants if px_peak_GA_mag[p]]

            else:
                peak_snrs[key] = [snr for p in self.participants for snr in px_peak_snrs[p]]
                peak_mags[key] = [mag for p in self.participants for mag in px_peak_mags[p]]
                peak_delays[key] = [delay for p in self.participants for delay in px_peak_delays[p]]
                peak_aucs[key] = [auc for p in self.participants for auc in px_peak_aucs[p]]
                peak_sigma[key] = [sigma for p in self.participants for sigma in px_peak_sigma[p]]
                peak_fwhm[key] = [fwhm for p in self.participants for fwhm in px_peak_fwhm[p]]
                peak_GA_mag[key] = [GA_mag for p in self.participants for GA_mag in px_peak_GA_mag[p]]

        if only_sig:
            ext = '_significant_trials_pP.npy' if per_participant else '_significant_trials.npy'
            np.save(self.files_per_condition_path.with_name(self.files_per_condition_path.stem + ext), significant_files_per_condition)

        results_dict = {"snr": peak_snrs, "mag": peak_mags, "delay": peak_delays, "auc": peak_aucs, "sigma": peak_sigma, "fwhm": peak_fwhm, "GAmag": peak_GA_mag}

        save_name = f'boxplot_data_crosscorr{"_onlysig" if only_sig else ""}{"_ga_peak" if ga_peak else ""}{"_fitted" if fit_peaks else ""}{"_avgP" if avg_participant else ""}{"_pP" if per_participant else ""}.npy'
        save_path = self.res_path / f'Boxplot_data_{subsample}'
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        np.save(save_path / save_name, results_dict, allow_pickle=True)
        
        return results_dict

    def prepare_boxplot_data_msc(self, corrected=False, per_participant=False, avg_participant=True):
        """Extract per-trial MSC metrics (band mean, band max, power-weighted band mean)
        per condition, optionally reduced to one value per participant, and save/return
        the result as boxplot-ready data.

        Parameters
        ----------
        corrected : bool, default False
            If True, use the noise-floor-corrected MSC values ('msc_band_mean_corrected',
            'msc_band_max_corrected') instead of the raw ones.
        per_participant : bool, default False
            If True, read each participant's per-participant grand-average MSC files (the
            'GA_msc' key) instead of single-trial files. Power-weighted mean is skipped in
            this case, since power-weighting is only meaningful for individual trials.
        avg_participant : bool, default True
            If True, average all of a participant's trials per condition first, so each
            participant contributes one value per condition. If False, keep one value per
            trial.

        Returns
        -------
        dict
            {"MSCmean": ..., "MSCmax": ..., "MSCpw": ...}, each value a dict mapping
            condition -> list of values (one per participant if `avg_participant`, else one
            per trial). Also saved to
            `res_path/Boxplot_data_msc(_corrected)/boxplot_data_msc...npy`.
        """

        corr = '_corrected'if corrected else ''
        prefix = 'GA_' if per_participant else ''

        msc_mean = {cond: [] for cond in self.conditions}
        msc_max = {cond: [] for cond in self.conditions}
        msc_pw = {cond: [] for cond in self.conditions}

        for cond in self.conditions:
            all_files = self.files_per_condition[cond][f'{prefix}msc']

            px_msc_mean = {p: [] for p in self.participants}
            px_msc_max = {p: [] for p in self.participants}
            px_msc_pw = {p: [] for p in self.participants} 

            for file in all_files:
                data = np.load(file, allow_pickle=True).item()
                p = participant_from_path(file)
                px_msc_mean[p].append(data[f'msc_band_mean{corr}'])
                px_msc_max[p].append(data[f'msc_band_max{corr}'])
                px_msc_pw[p].append(data[f'msc_band_mean{corr}_pw']) if not per_participant else None  # skip because power-weighting only makes sense for individual trials

            if avg_participant:
                msc_mean[cond] = [np.mean(px_msc_mean[p]) for p in self.participants if px_msc_mean[p]]
                msc_max[cond] = [np.mean(px_msc_max[p]) for p in self.participants if px_msc_max[p]]
                msc_pw[cond] = [np.mean(px_msc_pw[p]) for p in self.participants if px_msc_pw[p]]
            else:
                msc_mean[cond] = [msc for p in self.participants for msc in px_msc_mean[p]]
                msc_max[cond] = [msc for p in self.participants for msc in px_msc_max[p]]
                msc_pw[cond] = [msc for p in self.participants for msc in px_msc_pw[p]]

        results_dict = {"MSCmean": msc_mean, "MSCmax": msc_max, "MSCpw": msc_pw}

        save_name = f'boxplot_data_msc{corr}{"_avgP" if avg_participant else ""}{"_pP" if per_participant else ""}.npy'
        save_path = self.res_path / f'Boxplot_data_msc{corr}'
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        np.save(save_path / save_name, results_dict, allow_pickle=True)
        return results_dict

    def prepare_boxplot_data_psd(self, per_participant=False, avg_participant=True):
        """Extract per-trial in-band power (Pa² and dB) per condition, optionally reduced
        to one value per participant, and save/return the result as boxplot-ready data.

        Parameters
        ----------
        per_participant : bool, default False
            If True, read each participant's per-participant grand-average PSD files (the
            'GA_psd' key) instead of single-trial files.
        avg_participant : bool, default True
            If True, average all of a participant's trials per condition first, so each
            participant contributes one value per condition. If False, keep one value per
            trial.

        Returns
        -------
        dict
            {"PSD_Pa": ..., "PSD_dB": ...}, each value a dict mapping condition -> list of
            values (one per participant if `avg_participant`, else one per trial). Also
            saved to `res_path/Boxplot_data_psd/boxplot_data_psd...npy`.
        """
        prefix = 'GA_' if per_participant else ''

        psd_dB = {cond: [] for cond in self.conditions}
        psd_Pa = {cond: [] for cond in self.conditions}

        for cond in self.conditions:
            all_files = self.files_per_condition[cond][f'{prefix}psd']
            px_psd_dB = {p: [] for p in self.participants}
            px_psd_Pa = {p: [] for p in self.participants}

            for file in all_files:
                data = np.load(file, allow_pickle=True).item()
                p = participant_from_path(file)
                px_psd_dB[p].append(data['band_power_dB'])
                px_psd_Pa[p].append(data['band_power_Pa2'])

            if avg_participant:
                psd_dB[cond] = [np.mean(px_psd_dB[p]) for p in self.participants if px_psd_dB[p]]
                psd_Pa[cond] = [np.mean(px_psd_Pa[p]) for p in self.participants if px_psd_Pa[p]]
            else:
                psd_dB[cond] = [psd for p in self.participants for psd in px_psd_dB[p]]
                psd_Pa[cond] = [psd for p in self.participants for psd in px_psd_Pa[p]]

        results_dict = {"PSD_Pa": psd_Pa, "PSD_dB": psd_dB}

        save_name = f'boxplot_data_psd{"_avgP" if avg_participant else ""}{"_pP" if per_participant else ""}.npy'
        save_path = self.res_path / 'Boxplot_data_psd'
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        np.save(save_path / save_name, results_dict, allow_pickle=True)
        return results_dict

    def prepare_barplot_data_sigpeaks(self, per_participant=False, sig_border=None, peak_roi=25):
        """Count, per condition and per method (crosscorr, msc, psd), how many trials show
        a statistically significant peak versus the total number of trials available.

        Significance is determined per method by comparing the trial's peak value against
        the `self.sig_border` percentile of that trial's own noise-region values:
        cross-correlation envelope peak vs. envelope noise, MSC band max vs. the MSC values
        outside the frequency band, and PSD band power vs. the PSD values outside the band.

        Parameters
        ----------
        per_participant : bool, default False
            If True, use each participant's per-participant grand-average files (the
            'GA_crosscorr'/'GA_msc'/'GA_psd' keys) instead of single-trial files.
        sig_border : float, optional
            Percentile of the noise region a peak has to exceed. Default is `self.sig_border`.
        peak_roi : float, default 25
            Half-width in ms of the lag window searched for the cross-correlation peak.

        Returns
        -------
        dict
            {"no_sig_peaks": ..., "no_all_peaks": ...}, each value a dict
            {"crosscorr": {condition: count, ...}, "msc": ..., "psd": ...}. Also saved to
            `res_path/Barplot_data_sigpeaks/barplot_data_sigpeaks...npy`.
        """
        if sig_border is None:
            sig_border = self.sig_border

        prefix = 'GA_' if per_participant else ''

        inner_dict = {cond: 0 for cond in self.conditions}
        no_sig_peaks = {'crosscorr': inner_dict.copy(), 'msc': inner_dict.copy(), 'psd': inner_dict.copy()}
        no_all_peaks = {'crosscorr': inner_dict.copy(), 'msc': inner_dict.copy(), 'psd': inner_dict.copy()}

        # compute based on cross-correlation data
        for cond in self.conditions:
            all_files = self.files_per_condition[cond][f'{prefix}crosscorr']
            no_all_peaks['crosscorr'][cond] = len(all_files)

            for file in all_files:
                data = np.load(file)
                lag_ms = data[0, :] / self.fs * 1000
                env = data[3, :]

                _, env_noise = determine_noise_level(data)
                _, mag_peak = find_peak_amplitude(env, lag_ms, border=peak_roi)

                if mag_peak > np.percentile(env_noise, sig_border):
                    no_sig_peaks['crosscorr'][cond] += 1

        # compute based on msc data
        for cond in self.conditions:
            all_files = self.files_per_condition[cond][f'{prefix}msc']
            no_all_peaks['msc'][cond] = len(all_files)

            for file in all_files:
                data = np.load(file, allow_pickle=True).item()
                peak_val = data['msc_band_max']
                all_values = data['msc']
                band_mask = data['band_mask']  
                noise_region = all_values[~band_mask]
                if peak_val > np.percentile(noise_region, sig_border): 
                    no_sig_peaks['msc'][cond] += 1

        # compute based on psd data
        for cond in self.conditions:
            all_files = self.files_per_condition[cond][f'{prefix}psd']
            no_all_peaks['psd'][cond] = len(all_files)

            for file in all_files:
                data = np.load(file, allow_pickle=True).item()
                peak_val = data['band_power_dB']
                all_values = data['psd_dB']
                band_mask = data['band_mask']  
                noise_region = all_values[~band_mask]
                if np.abs(peak_val) > np.abs(np.percentile(noise_region, sig_border)): 
                    no_sig_peaks['psd'][cond] += 1

        results_dict = {"no_sig_peaks": no_sig_peaks, "no_all_peaks": no_all_peaks}

        save_name = f'barplot_data_sigpeaks{"_pP" if per_participant else ""}.npy'
        save_path = self.res_path / 'Barplot_data_sigpeaks'
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        np.save(save_path / save_name, results_dict, allow_pickle=True)
        return results_dict

    def convert_boxplot_data_for_amc(self, data_dict, save_name, dataset='JS26'):
        """Compute the Attention Modulation Coefficient (AMC) for each condition from boxplot
        data that distinguishes attention conditions.

        The AMC is calculated as: AMC = 2 * (Attended - Ignored) / (Attended + Ignored),
        applied element-wise per participant.

        Parameters
        ----------
        data_dict : dict
            Boxplot data of one metric (e.g. `prepare_boxplot_data_crosscorr(...)['snr']`),
            containing the keys '<harmonic condition> <voice> Attended' and
            '<harmonic condition> <voice> Ignored' (for MSA21 '<voice> Attended' etc.), each
            mapping to a list with one value per participant (i.e. computed with
            `avg_participant=True` or `per_participant=True`).
        save_name : str
            File name (including extension) under `res_path/Boxplot_data_amc/`.
        dataset : str, optional
            'JS25', 'JS26' or 'MSA21'; determines the harmonic conditions. Default is 'JS26'.

        Returns
        -------
        amc_results : dict
            {'<harmonic condition> <voice>': list of per-participant AMC values} (for MSA21
            {'<voice>': ...}).
        """
        harm_conds = HARMONIC_CONDITIONS[dataset]
        amc_results = {f'{g}': [] for g in ['M', 'F']} if dataset == 'MSA21' else {f'{pl} {g}': [] for pl in harm_conds for g in ['M', 'F']}

        for g in ['M', 'F']:
            for pl in harm_conds:
                subkey = f'{g}' if dataset == 'MSA21' else f'{pl} {g}'
                value_attended = np.array(data_dict[f'{subkey} Attended']) # list with one value per participant, so of length n_participants
                value_ignored = np.array(data_dict[f'{subkey} Ignored'])
                amc_results[subkey].extend(list(2 * (value_attended - value_ignored) / (value_attended + value_ignored)))
        
        save_path = self.res_path / 'Boxplot_data_amc'
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        np.save(save_path / save_name, amc_results, allow_pickle=True) 

        return amc_results
