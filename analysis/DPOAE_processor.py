"""Per-participant processing of speech-like DPOAE recordings.

`DPOAEProcessor` implements the complete per-participant analysis for the JS26 dataset
(competing-speaker experiment with pitch-shifted voices): loading of recordings and
experiment information, removal of the equipment delay, and evaluation of the speech-like
DPOAEs via cross-correlation, magnitude-squared coherence (MSC) and band power (PSD).

The dataset-specific subclasses `DPOAEProcessorJS25` and `DPOAEProcessorMSA21` override
only the methods that depend on the file layout of the respective dataset; all
signal-processing methods are shared.

Expected layout of a JS26 experiment folder (`exp_path`)::

    exp_path/
        data/                       raw data of all participants (recordings, PsychoPy csv
                                    files, <participant>_target_info.csv)
        Stimuli/                    harmonic waveforms, DP waveforms and presented stimuli
            STORY_Male/f0_to_140Hz/M_01_n6.wav, ...
            DP_STORY_Male/f0_to_140Hz/M_01_n4.wav, ...
            NOISE_Female/..., DP_NOISE_Female/..., ...
        results_<participant>/      created by this class
"""

import csv
import glob
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig

from functions_crosscorr import cross_corr
from functions_filter import butter_filter
from plot_grand_averages import plot_GA_crosscorr

# np.trapz was renamed to np.trapezoid in NumPy 2.0 (and later removed)
trapezoid = getattr(np, 'trapezoid', None) or np.trapz


class DPOAEProcessor:
    """Evaluate the speech-like DPOAE recordings of one participant of the JS26 dataset.

    Parameters
    ----------
    subject : str
        Participant ID, e.g. 'P01' or 'outofear_01'. Use 'Autocorrelations' for the
        participant-independent steps (`compute_autocorrelations`, `estimate_DPOAE_freqbands`).
    exp_path : str or Path
        Parent folder of the experiment (see the module docstring for its layout).
    res_path : str or Path, optional
        Folder the results are written to. Default is `exp_path/results_<subject>`.

    Attributes
    ----------
    data_path : Path
        Folder containing the raw data (`exp_path/data`).
    stim_path : Path
        Folder containing the harmonic and DP waveforms (`exp_path/Stimuli`).
    sig_border : int
        Percentile of the noise region that a peak has to exceed to count as significant.
    pref : float
        Reference sound pressure in Pa.
    fs : int
        Sampling frequency in Hz.
    gain : float
        Microphone amplifier gain in dB, subtracted from all levels.
    sens_mV : float
        Microphone sensitivity in V/Pa.
    harm_cond_map : dict
        Target f0 (in Hz) of the male and female voice for each harmonic condition.
    harm_number_map : dict
        Harmonic numbers [n1, n2] of the two stimulus components for each harmonic condition.
    """

    def __init__(self, subject, exp_path, res_path=None):
        self.subject = subject
        self.exp_path = Path(exp_path)
        self.data_path = self.exp_path / 'data'
        self.stim_path = self.exp_path / 'Stimuli'
        self.res_path = self.exp_path / f'results_{subject}' if res_path is None else Path(res_path)
        self.res_path.mkdir(parents=True, exist_ok=True)

        self.sig_border = 97  # percentile for significance
        self.pref = 20e-6  # reference pressure in Pa

        self.fs = 44100  # sampling frequency
        self.gain = 40  # microphone gain of 40 dB
        self.sens_mV = 50e-3  # microphone sensitivity in V/Pa

        self.harm_cond_map = {'low_res': {'Male': 140, 'Female': 180},
                              'low_unres': {'Male': 85, 'Female': 115},
                              'trans_res': {'Male': 270, 'Female': 325},
                              'trans_unres': {'Male': 120, 'Female': 160}}

        self.harm_number_map = {'low_res': [6, 8],
                                'low_unres': [10, 13],
                                'trans_res': [6, 8],
                                'trans_unres': [15, 18]}

    # ------------------------------------------------------------------
    # loading of experiment information and recordings
    # ------------------------------------------------------------------

    def load_experiment_info(self):
        """Load the responses to the comprehension questions and the listening-effort
        ratings from the csv file written by PsychoPy.

        Returns
        -------
        csv_info : pd.DataFrame
            One row per trial with the columns 'Rep', 'Given_Q1', 'Given_Q2', 'Given_Q3'
            and 'Listening_Effort'.
        """
        csv_files = sorted(glob.glob(str(self.data_path / f'{self.subject}_Setup_Objective1B_*.csv')))
        if not csv_files:
            raise FileNotFoundError(f'No PsychoPy csv file found for {self.subject} in {self.data_path}.')
        csv_info = pd.read_csv(csv_files[0])

        csv_cols = ['loop_attention_targets.thisRepN', 'loop_attention_targets.slider_Q1.response',
                    'loop_attention_targets.slider_Q2.response', 'loop_attention_targets.slider_Q3.response',
                    'loop_attention_targets.slider_ListeningEffort.response']
        csv_info = csv_info.loc[:, csv_cols]

        # PsychoPy writes additional rows for other routines; keep only the rows of the trial loop
        none_indices_base = csv_info.loc[:, csv_cols[0]].isna()
        for col in csv_cols[1:]:
            if not csv_info.loc[:, col].isna().equals(none_indices_base):
                raise ValueError(f'None indices do not match for column {col}!')
        csv_info = csv_info.loc[~none_indices_base, :]

        csv_info.columns = ['Rep', 'Given_Q1', 'Given_Q2', 'Given_Q3', 'Listening_Effort']
        return csv_info

    def load_target_info(self):
        """Load the randomised trial order (target voice, chapter, harmonic condition,
        stimulus and audiobook paths, correct answers) saved during the experiment.

        Returns
        -------
        target_info : pd.DataFrame
            Content of `data/<subject>_target_info.csv`.
        """
        return pd.read_csv(self.data_path / f'{self.subject}_target_info.csv', sep='\t')

    def get_given_answers(self, exp_info):
        """Extract the given answers to the three comprehension questions.

        Parameters
        ----------
        exp_info : pd.DataFrame
            Experiment information as returned by `load_experiment_info`.

        Returns
        -------
        answers : list of list of str
            The given answers ('a', 'b' or 'c') in the form [[Q1], [Q2], [Q3]], one entry
            per trial in each sublist.
        """
        answers = []
        for i in range(1, 4):
            answer = exp_info.loc[:, f'Given_Q{i}'].tolist()
            # slider positions 1, 2, 3 correspond to answers c, b, a
            answer = ['c' if int(item) == 1 else 'b' if int(item) == 2 else 'a' for item in answer]
            answers.append(answer)
        return answers

    def get_correct_answers(self, target_info):
        """Extract the correct answers to the three comprehension questions.

        Parameters
        ----------
        target_info : pd.DataFrame
            Target information as returned by `load_target_info`.

        Returns
        -------
        answers : list of list of str
            The correct answers in the form [[Q1], [Q2], [Q3]].
        """
        answers = []
        for i in range(1, 4):
            answer = target_info.loc[:, f'Correct A{i}'].tolist()
            answers.append([item.strip().strip("'") for item in answer])
        return answers

    def compare_answers(self, given, correct):
        """Compare given and correct answers to the comprehension questions.

        Parameters
        ----------
        given : list of list of str
            Given answers in the form [[Q1], [Q2], [Q3]].
        correct : list of list of str
            Correct answers in the same form.

        Returns
        -------
        correct_answers : list of list of int
            1 for a correct and 0 for a wrong answer, in the form [[Q1], [Q2], [Q3]].
        """
        return [[1 if given[i][j] == correct[i][j] else 0 for j in range(len(given[i]))] for i in range(3)]

    def get_listening_effort(self, exp_info):
        """Extract the listening-effort ratings.

        Parameters
        ----------
        exp_info : pd.DataFrame
            Experiment information as returned by `load_experiment_info`.

        Returns
        -------
        listening_effort : list
            One rating per trial.
        """
        return exp_info.loc[:, 'Listening_Effort'].to_numpy().flatten().tolist()

    def load_pure_tone_recs(self):
        """Load the recording of the pure-tone DPOAE measurement.

        Returns
        -------
        rec_pt : np.ndarray
            The pure-tone recording.
        """
        pt_recs = glob.glob(str(self.data_path / f'{self.subject}_**_recording_OAE_PT.wav'))
        if len(pt_recs) != 1:
            raise ValueError(f'Found {len(pt_recs)} pure tone recordings.')
        rec_pt, _ = librosa.load(pt_recs[0], sr=self.fs, mono=True, dtype=np.float64)
        return rec_pt

    def load_dpoae_recordings(self):
        """List the speech-like DPOAE recordings of the ear contralateral to the audiobook.

        Returns
        -------
        rec_paths : list of str
            Sorted paths to all recordings.
        """
        return sorted(glob.glob(str(self.data_path / f'{self.subject}_*_recordings_contralateral/*.wav')))

    def load_dp_waveforms(self, stim_path, return_dp_path=False):
        """Load the expected DP waveforms of the target (attended) and the noise (ignored)
        voice for a presented stimulus.

        Parameters
        ----------
        stim_path : str or Path
            Path of the presented stimulus as given in the target information, of the form
            '<folder>/<harm_cond>/competing_stimuli_STORY_<voice>_NOISE_<voice>/Ch_<ch>_....wav'.
            Harmonic condition, chapter and voices are read from this path.
        return_dp_path : bool, optional
            If True, return the paths of the DP waveforms instead of the waveforms. Default is False.

        Returns
        -------
        stim_name : str
            Pairing of story and noise voice, e.g. 'STORY_Male_NOISE_Female'.
        wf_dp_attended : np.ndarray or Path
            DP waveform of the target voice (or its path).
        wf_dp_ignored : np.ndarray or Path
            DP waveform of the noise voice (or its path).
        """
        target_info = str(stim_path).split('/')[-2].split('_')
        target_voice, noise_voice = target_info[3], target_info[-1]
        target_folder = f'{target_info[2]}_{target_voice}'
        noise_folder = f'{target_info[-2]}_{noise_voice}'

        harm_cond = str(stim_path).split('/')[1]
        chapter = str(stim_path).split('/')[-1].split('_')[1]
        ndp = 2 * self.harm_number_map[harm_cond][0] - self.harm_number_map[harm_cond][1]

        target_path = (self.stim_path / f'DP_{target_folder}' / f'f0_to_{self.harm_cond_map[harm_cond][target_voice]}Hz'
                       / f'{target_voice[0]}_{chapter}_n{ndp}.wav')
        noise_path = (self.stim_path / f'DP_{noise_folder}' / f'f0_to_{self.harm_cond_map[harm_cond][noise_voice]}Hz'
                      / f'NOISE_{noise_voice[0]}_{chapter}_n{ndp}.wav')

        stim_name = '_'.join([target_folder, noise_folder])

        if return_dp_path:
            return stim_name, target_path, noise_path

        wf_dp_attended = librosa.load(target_path, sr=self.fs, mono=True, dtype=np.float64)[0]
        wf_dp_ignored = librosa.load(noise_path, sr=self.fs, mono=True, dtype=np.float64)[0]
        return stim_name, wf_dp_attended, wf_dp_ignored

    def load_dpoae_bandwidths(self):
        """Load the DPOAE frequency bands, estimating them first if they have not been saved yet.

        Returns
        -------
        dict
            Frequency bands as returned by `estimate_DPOAE_freqbands`.
        """
        bandwidths_path = self.exp_path / 'dpoae_bandwidths.npy'
        if not bandwidths_path.exists():
            self.estimate_DPOAE_freqbands()
        return dict(np.load(bandwidths_path, allow_pickle=True).item())

    def write_results_csv(self, columns, results, save_path):
        """Write a table of results to a csv file.

        Parameters
        ----------
        columns : list of str
            Column names.
        results : list of list
            One list per row; the number of entries must match the number of columns.
        save_path : str or Path
            Path of the csv file.
        """
        with open(save_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(results)

    # ------------------------------------------------------------------
    # levels and frequencies
    # ------------------------------------------------------------------

    def volts_per_full_scale(self, input_dbu_at_full_scale=19, input_digital_gain_db=0):
        """Voltage at the input of the audio interface (RME Fireface) that corresponds to a
        sample value of 1.0.

        0 dBFS is the RMS level of a full-scale sine, so the peak (sample value 1.0) is
        sqrt(2) higher. 0 dBu = 0.7746 V RMS; at the Lo Gain setting, +19 dBu correspond to
        6.90 V RMS and hence 9.76 V per 1.0.

        Parameters
        ----------
        input_dbu_at_full_scale : float, optional
            Input level in dBu that corresponds to 0 dBFS. Default is 19 (Lo Gain).
        input_digital_gain_db : float, optional
            Additional digital input gain in dB. Default is 0.

        Returns
        -------
        float
            Voltage in V per sample value of 1.0.
        """
        vrms_full_scale = 0.7746 * 10 ** (input_dbu_at_full_scale / 20)
        return vrms_full_scale * np.sqrt(2) / 10 ** (input_digital_gain_db / 20)

    def estimate_level(self, rec, gate_dB=-40, win_ms=50):
        """Estimate the sound level of a recording in dB SPL.

        The level is the RMS over all samples whose short-term RMS (moving window of
        `win_ms`) lies within `gate_dB` of the loudest window, so that silent passages do
        not lower the estimate.

        Parameters
        ----------
        rec : np.ndarray
            Recording in sample values.
        gate_dB : float, optional
            Gating threshold in dB relative to the loudest window. Default is -40 dB.
        win_ms : float, optional
            Window length of the short-term RMS in ms. Default is 50 ms.

        Returns
        -------
        dB : float
            Gated RMS level in dB SPL (-inf if the recording is silent).
        max_rms_dB : float
            Level of the loudest window in dB SPL (-inf if the recording is silent).
        """
        pressure = rec * self.volts_per_full_scale() / self.sens_mV  # in Pa

        # moving RMS via convolution
        win_len = max(1, int(self.fs * win_ms / 1000))
        window = np.ones(win_len) / win_len
        rms_env = np.sqrt(np.convolve(pressure.astype(np.float64) ** 2, window, mode='same'))

        max_rms = np.max(rms_env)
        if max_rms == 0:
            return -np.inf, -np.inf  # no signal

        active = rms_env > max_rms * (10 ** (gate_dB / 20))
        if not np.any(active):
            return -np.inf, -np.inf

        rms_active = np.sqrt(np.mean(pressure[active].astype(np.float64) ** 2))
        dB = 20 * np.log10(rms_active / self.pref) - self.gain
        max_rms_dB = 20 * np.log10(max_rms / self.pref) - self.gain
        return dB, max_rms_dB

    def estimate_stim_frequency(self, signal, return_median=True, return_min_max=False):
        """Estimate the frequency of a (harmonic or DP) waveform from its instantaneous
        frequency, computed via the Hilbert transform after band-pass filtering (100-5000 Hz).
        Outliers are removed with an interquartile-range criterion.

        Parameters
        ----------
        signal : np.ndarray
            The waveform.
        return_median : bool, optional
            If True, return median and standard deviation of the instantaneous frequency.
            Default is True.
        return_min_max : bool, optional
            Only used if `return_median` is False. If True, additionally return the
            frequencies bounding the central 90 % of the spectral energy (Welch PSD).
            Default is False.

        Returns
        -------
        tuple or np.ndarray
            (median, std) if `return_median`; (median, std, f_low, f_high) if
            `return_min_max`; otherwise the instantaneous-frequency array.
        """
        sos = sig.butter(4, [100, 5000], btype='bandpass', fs=self.fs, output='sos')
        signal_filt = sig.sosfilt(sos, signal)

        phase = np.unwrap(np.angle(sig.hilbert(signal_filt)))
        inst_freq = np.diff(phase) * self.fs / (2 * np.pi)

        # IQR-based outlier removal (more robust than bare frequency bounds)
        q1, q3 = np.percentile(inst_freq, [25, 75])
        iqr = q3 - q1
        fence_lo = max(q1 - 3 * iqr, 0)
        fence_hi = min(q3 + 3 * iqr, self.fs / 2)
        inst_freq_clean = inst_freq[(inst_freq >= fence_lo) & (inst_freq <= fence_hi)]

        median_freq = np.median(inst_freq_clean)
        std_freq = np.std(inst_freq_clean)

        if return_median:
            return median_freq, std_freq
        if return_min_max:
            nperseg = min(2048, len(signal_filt) // 4)
            freqs, psd = sig.welch(signal_filt, fs=self.fs, nperseg=nperseg, noverlap=nperseg // 2,
                                   window='hann', scaling='density')
            cumulative_energy = np.cumsum(psd)
            cumulative_energy /= cumulative_energy[-1]
            f_low = freqs[np.searchsorted(cumulative_energy, 0.05)]
            f_high = freqs[np.searchsorted(cumulative_energy, 0.95)]
            return median_freq, std_freq, f_low, f_high
        return inst_freq

    def estimate_DPOAE_freqbands(self):
        """Estimate the frequency band of the DPOAE for each harmonic condition and voice.

        For every DP waveform, the band containing the central 90 % of its spectral energy
        is determined (`estimate_stim_frequency`). The band limits are averaged across
        chapters and across story and noise waveforms of the same voice. The result is
        saved to `exp_path/dpoae_bandwidths.npy` and used for the MSC and PSD analyses.

        Returns
        -------
        dict
            {harm_cond: {'Male': [f_low, f_high], 'Female': [f_low, f_high]}} in Hz.
        """
        dpoae_bandwidths = {cond: {} for cond in self.harm_cond_map}

        for folder_path in sorted(self.stim_path.glob('DP_*')):
            name = folder_path.name  # e.g. 'DP_STORY_Male'
            for harm_cond in self.harm_cond_map:
                shift = self.harm_cond_map[harm_cond][name.split('_')[-1]]
                ndp = 2 * self.harm_number_map[harm_cond][0] - self.harm_number_map[harm_cond][1]
                all_dp_files = glob.glob(str(folder_path / f'f0_to_{shift}Hz' / f'*n{ndp}.wav'))

                results_low, results_high = [], []
                for file in all_dp_files:
                    dp_wf = librosa.load(file, sr=self.fs, mono=True)[0]
                    _, _, f_low, f_high = self.estimate_stim_frequency(dp_wf, return_median=False, return_min_max=True)
                    results_low.append(f_low)
                    results_high.append(f_high)
                dpoae_bandwidths[harm_cond][name] = [np.mean(results_low), np.mean(results_high)]

        dpoae_bandwidths_voice = {cond: {v: [] for v in ['Male', 'Female']} for cond in self.harm_cond_map}
        for voice in ['Male', 'Female']:
            for harm_cond in self.harm_cond_map:
                val_story = dpoae_bandwidths[harm_cond][f'DP_STORY_{voice}']
                val_noise = dpoae_bandwidths[harm_cond][f'DP_NOISE_{voice}']
                dpoae_bandwidths_voice[harm_cond][voice].extend(np.mean([val_story, val_noise], axis=0))

        np.save(self.exp_path / 'dpoae_bandwidths.npy', dpoae_bandwidths_voice, allow_pickle=True)
        return dpoae_bandwidths_voice

    # ------------------------------------------------------------------
    # DPOAE metrics
    # ------------------------------------------------------------------

    def compute_PT_PSD(self):
        """Compute the power spectral density of the pure-tone recording and save a plot
        (`<subject>_PSD_PT.png`).

        Returns
        -------
        freq : np.ndarray
            Frequencies in Hz.
        psd : np.ndarray
            Power per frequency bin in Pa².
        psd_dB : np.ndarray
            Power spectral density in dB SPL/Hz.
        """
        x = self.load_pure_tone_recs() / self.sens_mV  # convert to Pa
        n = len(x)

        fftx = np.fft.fft(x)[:n // 2]
        freq = np.fft.fftfreq(n, 1 / self.fs)[:n // 2]

        psd = np.abs(fftx) ** 2 / n ** 2  # in Pa^2
        psd_per_bin = psd / (self.fs / n)  # in Pa^2/Hz
        psd_dB = 10 * np.log10(psd_per_bin / self.pref ** 2) - self.gain  # in dB SPL/Hz

        plt.figure()
        plt.plot(freq, psd_dB)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power/Frequency [dB SPL/Hz]')
        plt.title('Power Spectral Density')
        plt.xlim(600, 1600)
        plt.ylim(-60, 100)
        plt.grid()
        plt.tight_layout()
        plt.savefig(self.res_path / f'{self.subject}_PSD_PT.png')
        plt.close()

        return freq, psd, psd_dB

    def compute_autocorrelations(self):
        """Compute the autocorrelation of every DP waveform and their average per harmonic
        condition, voice and role (story = attended, noise = ignored).

        The averaged autocorrelations are used by `DPOAEStatistics` to initialise the
        Gaussian fits of the grand-average cross-correlations. Results are saved under
        `res_path` (one folder per DP waveform folder), and all averages together in
        `res_path/GA_autocorr_dict.npz` with keys such as 'Low_res M Attended'.
        """
        for sub in sorted(p for p in self.stim_path.glob('DP_*') if p.is_dir()):
            voice = sub.name.split('_')[-1]
            for harm_type in self.harm_cond_map:
                shift = self.harm_cond_map[harm_type][voice]
                harms = self.harm_number_map[harm_type]
                ndp = 2 * harms[0] - harms[1]
                all_files = glob.glob(str(sub / f'f0_to_{shift}Hz' / f'*_n{ndp}.wav'))
                save_path = self.res_path / sub.name / harm_type
                save_path.mkdir(parents=True, exist_ok=True)

                auto_ga = np.array([])
                for file in all_files:
                    dp_stim, _ = librosa.load(file, sr=self.fs, mono=True)
                    auto_lag, auto_real, auto_imag, auto_env = cross_corr(dp_stim, dp_stim)
                    auto_stats = np.array([auto_lag, auto_real, auto_imag, auto_env])
                    auto_ga = np.append(auto_ga, auto_stats)
                    np.save(save_path / f'autocorr_{Path(file).stem}.npy', auto_stats)

                    lag_peak, mag_peak, _, _ = self.find_significant_peak(auto_lag, auto_env)
                    self.plot_DPOAE_crosscorr(auto_lag, auto_real, auto_imag, auto_env, lag_peak, mag_peak,
                                              save_path / f'autocorr_{Path(file).stem}.png')

                auto_ga_result = np.mean(auto_ga.reshape(len(all_files), 4, -1), axis=0)
                plot_GA_crosscorr(f'Autocorrelation {harm_type}', auto_ga_result, save_path / f'GA_autocorr_{harm_type}.png')
                np.save(save_path / f'GA_autocorr_{harm_type}.npy', auto_ga_result)

        all_GA_files = glob.glob(str(self.res_path / '**/GA_autocorr_*.npy'), recursive=True)
        if len(all_GA_files) > 16:
            raise ValueError(f'Found {len(all_GA_files)} GA files, but expected at most 16!')

        GA_dict = {}
        for file in all_GA_files:
            parts = Path(file).parts
            harm_type = parts[-2]
            voice = parts[-3].split('_')[-1]
            attention = 'Attended' if 'STORY' in parts[-3] else 'Ignored'
            GA_dict[f'{harm_type.capitalize()} {voice[0]} {attention}'] = np.load(file)

        np.savez_compressed(self.res_path / 'GA_autocorr_dict.npz', **GA_dict)

    def compute_equipment_delay(self, stim_orig, rec, save_path=None, figure_format='png', smooth_env_cutoff=8):
        """Determine the delay of the recording equipment from the cross-correlation of the
        recording with the presented stimulus waveforms.

        Parameters
        ----------
        stim_orig : list of np.ndarray
            The presented waveforms (the two stimulus channels).
        rec : np.ndarray
            The microphone recording.
        save_path : Path, optional
            Path (without extension) of a diagnostic plot. No plot is saved if None.
        figure_format : str, optional
            'png' or 'svg'. Default is 'png'.
        smooth_env_cutoff : float, optional
            Smoothing factor passed to `smooth_envelope`. Default is 8.

        Returns
        -------
        eq_delay : float
            Equipment delay in samples (mean of the delays of all waveforms, rounded down).
        """
        disp_border = 40  # ms
        delay = np.zeros(len(stim_orig))
        plt.figure(figsize=(10, 7))

        for i, stim in enumerate(stim_orig):
            if len(stim) > len(rec):
                stim = stim[:len(rec)]
            elif len(stim) < len(rec):
                rec = rec[:len(stim)]

            lag, corr_real, corr_imag, env = cross_corr(rec, stim, fs=self.fs)
            env = self.smooth_envelope(corr_real, env, cf=smooth_env_cutoff)
            delay[i] = lag[np.argmax(env)]
            mag = np.max(env)

            lag_ms = (lag - delay[i]) / self.fs * 1000
            plt.subplot(2, 1, i + 1)
            plt.plot(lag_ms, corr_real, label='real')
            plt.plot(lag_ms, corr_imag, label='imag')
            plt.plot(lag_ms, env, label='envelope', c='k')
            plt.plot(lag_ms, -env, c='k')
            plt.legend(loc='upper right', ncol=2)
            plt.xlabel('Lag [ms]')
            plt.ylabel('Correlation')
            plt.title(f'Correlation of f{i + 1} vs. microphone recording')
            plt.grid()
            plt.xlim(-disp_border, disp_border)
            plt.ylim(-1.3 * mag, 1.3 * mag)
            plt.tight_layout()

        if save_path is not None:
            if figure_format == 'png':
                plt.savefig(save_path.with_suffix('.png'), format='png')
            elif figure_format == 'svg':
                plt.savefig(save_path.with_suffix('.svg'), format='svg', dpi=1200, transparent=True)
            else:
                raise ValueError('Invalid file format! Must be png or svg.')
        plt.close()

        return np.floor(np.mean(delay))

    def compute_DPOAE_crosscorr(self, dp, rec, eq_delay, save_path, figure_format='.png', smooth_env_cutoff=10,
                                peak_roi=15, disp_border=60):
        """Cross-correlate the recording with the expected DP waveform and evaluate the peak.

        The complex cross-correlation (`cross_corr`) is computed, its envelope optionally
        smoothed, the lag axis corrected for the equipment delay and cut to ±0.5 s
        (`center_crosscorr`), and the largest envelope peak within ±`peak_roi` ms tested for
        significance (`find_significant_peak`). The centred cross-correlation is saved as
        `<save_path>corr.npy` in the form [lag, real, imag, envelope], and plotted.

        Parameters
        ----------
        dp : np.ndarray
            Expected DP waveform; must have the same length as `rec`.
        rec : np.ndarray
            Microphone recording.
        eq_delay : int
            Equipment delay in samples, subtracted from the lag axis.
        save_path : Path
            Prefix of the output files, e.g. `.../Attended_DPs/Ch1_`.
        figure_format : str, optional
            '.png' or '.svg'. Default is '.png'.
        smooth_env_cutoff : float or None, optional
            Smoothing factor passed to `smooth_envelope`; no smoothing if None. Default is 10.
        peak_roi : float, optional
            Half-width in ms of the lag window searched for the DPOAE peak. Default is 15 ms.
        disp_border : float, optional
            Half-width in ms of the zoomed plot. Default is 60 ms.

        Returns
        -------
        stats : list
            [lag of the peak (samples), peak magnitude, SNR (dB), significant (bool)].
        """
        eq_delay = int(eq_delay)
        assert len(dp) == len(rec), 'The length of the DP waveform and the recording must be the same.'

        lag, real, imag, env = cross_corr(rec, dp, fs=self.fs, limit=False)
        if smooth_env_cutoff is not None:
            env = self.smooth_envelope(real, env, cf=smooth_env_cutoff)

        lag = lag - eq_delay
        lag_c, real_c, imag_c, env_c = self.center_crosscorr(lag, real, imag, env)
        lag_peak, mag_peak, is_sig, snr = self.find_significant_peak(lag_c, env_c, roi=peak_roi)

        self.plot_DPOAE_crosscorr(lag_c, real_c, imag_c, env_c, lag_peak, mag_peak,
                                  save_path.with_name(save_path.name + 'DP'), figure_format=figure_format,
                                  disp_border=disp_border)
        np.save(save_path.with_name(save_path.name + 'corr.npy'), np.array([lag_c, real_c, imag_c, env_c]))

        return [lag_peak, mag_peak, snr, is_sig]

    def compute_DPOAE_PSD(self, dp_name, rec, band, save_path=None):
        """Compute the power of the recording within the DPOAE frequency band (Welch PSD).

        Parameters
        ----------
        dp_name : str
            Label used in the plot title and the csv output.
        rec : np.ndarray
            Microphone recording in V.
        band : sequence of float
            (f_low, f_high) of the DPOAE band in Hz (see `estimate_DPOAE_freqbands`).
        save_path : Path, optional
            Prefix of the output files (`<save_path>power_spectrum.png`,
            `<save_path>PSD_values.csv`). Nothing is saved if None.

        Returns
        -------
        dict
            'band_power_Pa2', 'band_power_dB' (dB SPL), 'freqs', 'psd_Pa2_per_Hz',
            'psd_dB' (dB SPL/Hz), 'band_mask', 'df_Hz' and 'n_averages'.
        """
        rec_Pa = rec / self.sens_mV  # convert to Pa

        # 1-second segments or 1/8 of the recording, whichever is smaller; at least 512 samples
        nperseg = max(512, min(int(self.fs), len(rec_Pa) // 8))
        noverlap = nperseg // 2

        freqs, psd = sig.welch(rec_Pa, fs=self.fs, nperseg=nperseg, noverlap=noverlap,
                               window='hann', scaling='density', return_onesided=True)

        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        if not band_mask.any():
            raise ValueError(f'Band {band} Hz contains no Welch frequency bins. '
                             f'Frequency resolution is {self.fs / nperseg:.2f} Hz; widen the band or increase nperseg.')
        band_power_Pa2 = trapezoid(psd[band_mask], freqs[band_mask])
        band_power_dB = 10 * np.log10(band_power_Pa2 / self.pref ** 2) - self.gain

        psd_safe = np.where(psd > 0, psd, np.nan)  # avoid log(0)
        psd_dB = 10 * np.log10(psd_safe / self.pref ** 2) - self.gain

        df = freqs[1] - freqs[0]
        n_averages = 1 + (len(rec_Pa) - nperseg) // (nperseg - noverlap)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(freqs, psd_dB, lw=0.8, label='Welch PSD')
        ax.axvline(band[0], color='r', linestyle='--', lw=1.2, label=f'Band start ({band[0]:.0f} Hz)')
        ax.axvline(band[1], color='g', linestyle='--', lw=1.2, label=f'Band end ({band[1]:.0f} Hz)')
        ax.fill_between(freqs, psd_dB, where=band_mask, alpha=0.25, color='orange',
                        label=f'Band power = {band_power_dB:.1f} dB SPL')
        ax.set_xlim(0, 3000)
        ax.set_ylim(-50, 70)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('PSD [dB SPL/Hz]')
        ax.set_title(f'Band Power — {dp_name}\n nperseg={nperseg}, df={df:.2f} Hz, ~{n_averages} averages')
        ax.grid(alpha=0.4)
        ax.legend(fontsize=8)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path.with_name(save_path.name + 'power_spectrum.png'))
        plt.close(fig)

        if save_path is not None:
            col_names = ['DP Name', 'fmin [Hz]', 'fmax [Hz]', 'nperseg', 'df [Hz]', 'n_averages',
                         'Band Power [Pa^2]', 'Band Power [dB SPL]']
            results = [dp_name, band[0], band[1], nperseg, round(df, 4), n_averages, band_power_Pa2, band_power_dB]
            pd.DataFrame([results], columns=col_names).to_csv(save_path.with_name(save_path.name + 'PSD_values.csv'),
                                                              index=False)

        return {'band_power_Pa2': band_power_Pa2, 'band_power_dB': band_power_dB, 'freqs': freqs,
                'psd_Pa2_per_Hz': psd, 'psd_dB': psd_dB, 'band_mask': band_mask, 'df_Hz': df,
                'n_averages': n_averages}

    def compute_DPOAE_MSC(self, template, recording, band, nperseg=16384):
        """Compute the magnitude-squared coherence (MSC) between the expected DP waveform and
        the recording, with bias correction and averaging within the DPOAE band.

        Parameters
        ----------
        template : np.ndarray
            Expected DP waveform; must have the same length as `recording`.
        recording : np.ndarray
            Microphone recording.
        band : sequence of float
            (f_low, f_high) of the DPOAE band in Hz.
        nperseg : int, optional
            Segment length of Welch's method. Default is 16384 samples (~371 ms, ~2.7 Hz
            resolution at 44.1 kHz).

        Returns
        -------
        dict
            'freqs', 'msc', 'msc_corrected' (bias-corrected, see below), band mean and maximum
            of both ('msc_band_mean', 'msc_band_max', 'msc_band_mean_corrected',
            'msc_band_max_corrected'), band means weighted by the template's power spectrum
            ('msc_band_mean_pw', 'msc_band_mean_corrected_pw'), 'n_segments' (K),
            'expected_msc_H0' (1/K), 'band_mask', 'nperseg' and 'df'.

        Notes
        -----
        Under the null hypothesis of no coherence, the expected MSC estimate is 1/K for K
        averaged segments. The corrected MSC maps [1/K, 1] linearly to [0, 1]; values below
        1/K are clipped to 0.
        """
        n = len(template)
        assert n == len(recording), 'Template and recording must have the same length.'

        noverlap = nperseg // 2
        step = nperseg - noverlap

        freqs, msc = sig.coherence(template, recording, fs=self.fs, nperseg=nperseg, noverlap=noverlap, window='hann')
        _, S_xx = sig.welch(template, fs=self.fs, nperseg=nperseg, noverlap=noverlap, window='hann', scaling='density')

        K = 1 + (n - nperseg) // step  # number of Welch segments
        msc_corrected = np.clip((K * msc - 1) / (K - 1), 0, 1)

        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        if not band_mask.any():
            raise ValueError(f'Band {band} Hz contains no frequency bins. Resolution is {self.fs / nperseg:.2f} Hz/bin. '
                             f'Increase nperseg or widen the band.')

        weights = S_xx[band_mask] / np.sum(S_xx[band_mask])
        msc_band = msc[band_mask]
        msc_corr_band = msc_corrected[band_mask]

        return {'freqs': freqs, 'msc': msc, 'msc_corrected': msc_corrected,
                'msc_band_mean': msc_band.mean(), 'msc_band_max': msc_band.max(),
                'msc_band_mean_corrected': msc_corr_band.mean(), 'msc_band_max_corrected': msc_corr_band.max(),
                'msc_band_mean_pw': np.dot(weights, msc_band), 'msc_band_mean_corrected_pw': np.dot(weights, msc_corr_band),
                'n_segments': K, 'expected_msc_H0': 1.0 / K, 'band_mask': band_mask, 'nperseg': nperseg,
                'df': freqs[1] - freqs[0]}

    def center_crosscorr(self, lag, real, imag, env, border_s=0.5):
        """Cut a cross-correlation to the lags within ±`border_s` around zero.

        Parameters
        ----------
        lag : np.ndarray
            Lags in samples (already corrected for the equipment delay).
        real, imag, env : np.ndarray
            Real part, imaginary part and envelope of the cross-correlation.
        border_s : float, optional
            Half-width of the retained segment in s. Default is 0.5 s.

        Returns
        -------
        lag, real, imag, env : np.ndarray
            The cut arrays.
        """
        border = int(border_s * self.fs)
        idx_min = np.where(lag == -border)[0][0]
        idx_max = np.where(lag == border)[0][0]
        sl = slice(idx_min, idx_max + 1)
        return lag[sl], real[sl], imag[sl], env[sl]

    def smooth_envelope(self, carrier, env, cf=8):
        """Smooth the envelope of a cross-correlation with a low-pass filter whose cutoff is
        the dominant frequency of the carrier (real part) divided by `cf`.

        Parameters
        ----------
        carrier : np.ndarray
            Real part of the cross-correlation.
        env : np.ndarray
            Envelope of the cross-correlation.
        cf : float, optional
            Divisor of the dominant carrier frequency; larger values smooth more. Default is 8.

        Returns
        -------
        env : np.ndarray
            The smoothed envelope (unchanged if the cutoff would be 0 Hz).
        """
        fft_real = np.abs(np.fft.fft(carrier))
        fft_real = fft_real[:len(fft_real) // 2][1:]  # positive frequencies without DC
        freq_real = np.fft.fftfreq(len(carrier), 1 / self.fs)
        cutoff = freq_real[np.argmax(fft_real)] / cf

        if cutoff > 0:
            env = butter_filter(env, cutoff, 'lowpass')
        return env

    def find_significant_peak(self, lag, env, roi=15):
        """Find the largest envelope peak within ±`roi` ms and test it against the noise region.

        The noise region comprises the lags between 70 and 750 ms on both sides. A peak is
        significant if it exceeds the `sig_border` percentile of the noise region.

        Parameters
        ----------
        lag : np.ndarray
            Lags in samples.
        env : np.ndarray
            Envelope of the cross-correlation.
        roi : float, optional
            Half-width of the search window in ms. Default is 15 ms.

        Returns
        -------
        lag_peak : float
            Lag of the peak in samples.
        mag_peak : float
            Peak magnitude.
        is_sig : bool
            Whether the peak is significant.
        snr : float
            Peak magnitude relative to the mean of the noise region, in dB.
        """
        border = roi / 1000 * self.fs
        idx_min = np.argmin(np.abs(lag + border))
        idx_max = np.argmin(np.abs(lag - border))
        env_range = env[idx_min:idx_max + 1]

        mag_peak = np.max(env_range)
        lag_peak = lag[np.argmax(env_range) + idx_min]

        l, h = 70 / 1000 * self.fs, 750 / 1000 * self.fs
        env_noise = env[np.where(((lag < -l) & (lag > -h)) | ((lag > l) & (lag < h)))[0]]

        is_sig = bool(mag_peak > np.percentile(env_noise, self.sig_border))
        snr = 20 * np.log10(mag_peak / np.mean(env_noise))
        return lag_peak, mag_peak, is_sig, snr

    # ------------------------------------------------------------------
    # per-participant grand averages
    # ------------------------------------------------------------------

    def _leaf_directories(self, parent_path=None):
        """Return all folders below `parent_path` (default `res_path`) that contain no subfolders."""
        root = self.res_path if parent_path is None else Path(parent_path)
        return [p for p in root.rglob('*') if p.is_dir() and not any(child.is_dir() for child in p.iterdir())]

    def average_crosscorr_per_condition(self, parent_path=None):
        """Average the single-trial cross-correlations in every condition folder and save the
        participant's grand average (`GA_data.npy`) and a plot (`GA.png`).

        Parameters
        ----------
        parent_path : str or Path, optional
            Folder to search for condition folders. Default is `res_path`.
        """
        for sub in self._leaf_directories(parent_path):
            corr_files = sorted(sub.glob('*corr.npy'))
            if not corr_files:
                continue
            folder_GA = np.concatenate([np.load(corr).ravel() for corr in corr_files])
            GA_result = np.mean(folder_GA.reshape(len(corr_files), 4, -1), axis=0)
            plot_GA_crosscorr('', GA_result, sub / 'GA.png')
            np.save(sub / 'GA_data.npy', GA_result)

    def average_MSC_per_condition(self, parent_path=None):
        """Average the single-trial MSC spectra in every condition folder and save the
        participant's grand average (`GA_MSC_data.npy`, same keys as `compute_DPOAE_MSC`
        except the power-weighted means) and a plot (`GA_MSC.png`).

        Parameters
        ----------
        parent_path : str or Path, optional
            Folder to search for condition folders. Default is `res_path`.
        """
        for sub in self._leaf_directories(parent_path):
            msc_files = sorted(sub.glob('*MSC_results.npy'))
            if not msc_files:
                continue
            msc_list, msc_corr_list = [], []
            for msc in msc_files:
                data = np.load(msc, allow_pickle=True).item()
                msc_list.append(data['msc'])
                msc_corr_list.append(data['msc_corrected'])

            mean_GA = np.mean(np.vstack(msc_list), axis=0)
            mean_GA_corrected = np.mean(np.vstack(msc_corr_list), axis=0)
            band_mask = data['band_mask']

            GA_result = {'freqs': data['freqs'], 'msc': mean_GA, 'msc_corrected': mean_GA_corrected,
                         'msc_band_mean': mean_GA[band_mask].mean(), 'msc_band_max': mean_GA[band_mask].max(),
                         'msc_band_mean_corrected': mean_GA_corrected[band_mask].mean(),
                         'msc_band_max_corrected': mean_GA_corrected[band_mask].max(),
                         'n_segments': data['n_segments'], 'band_mask': band_mask, 'nperseg': data['nperseg'],
                         'df': data['df']}

            self.plot_DPOAE_MSC(GA_result, title=f'GA MSC {sub.parts[-3]} {sub.parts[-2]}', show_corrected=True,
                                save_path=sub / 'GA_MSC.png')
            np.save(sub / 'GA_MSC_data.npy', GA_result)

    def average_PSD_per_condition(self, parent_path=None):
        """Average the single-trial power spectra in every condition folder and save the
        participant's grand average with its band power (`GA_PSD_data.npy`, same keys as
        `compute_DPOAE_PSD`).

        Parameters
        ----------
        parent_path : str or Path, optional
            Folder to search for condition folders. Default is `res_path`.
        """
        for sub in self._leaf_directories(parent_path):
            psd_files = sorted(sub.glob('*PSD_results.npy'))
            if not psd_files:
                continue
            ex_chapter = str(psd_files[0].name).split('_')[0]
            band_info = pd.read_csv(psd_files[0].with_name(f'{ex_chapter}_PSD_values.csv'))[['fmin [Hz]', 'fmax [Hz]']].values[0]

            psd_Pa2_list, psd_dB_list = [], []
            for psd_file in psd_files:
                data = np.load(psd_file, allow_pickle=True).item()
                psd_Pa2_list.append(data['psd_Pa2_per_Hz'])
                psd_dB_list.append(data['psd_dB'])

            band_mask = (data['freqs'] >= np.floor(band_info[0])) & (data['freqs'] <= np.ceil(band_info[1]))
            GA_Pa2 = np.mean(np.vstack(psd_Pa2_list), axis=0)
            GA_dB = np.mean(np.vstack(psd_dB_list), axis=0)
            GA_band_power_Pa2 = trapezoid(GA_Pa2[band_mask], data['freqs'][band_mask])

            GA_result = {'band_power_Pa2': GA_band_power_Pa2,
                         'band_power_dB': 10 * np.log10(GA_band_power_Pa2 / self.pref ** 2) - self.gain,
                         'freqs': data['freqs'], 'psd_Pa2_per_Hz': GA_Pa2, 'psd_dB': GA_dB, 'band_mask': band_mask,
                         'df_Hz': data['df_Hz'], 'n_averages': data['n_averages']}
            np.save(sub / 'GA_PSD_data.npy', GA_result)

    # ------------------------------------------------------------------
    # plots
    # ------------------------------------------------------------------

    def plot_DPOAE_crosscorr(self, lag, corr_real, corr_imag, env, lag_peak, mag, save_path=None, limit=True,
                             figure_format='.png', disp_border=60):
        """Plot a cross-correlation between DP waveform and recording.

        Saves `<save_path>_complete` (all lags) and, if `limit`, `<save_path>_limits`
        (±`disp_border` ms).

        Parameters
        ----------
        lag : np.ndarray
            Lags in samples.
        corr_real, corr_imag, env : np.ndarray
            Real part, imaginary part and envelope of the cross-correlation.
        lag_peak : float
            Lag of the peak in samples (marked by a dashed line).
        mag : float
            Peak magnitude (sets the y-limits of the zoomed plot).
        save_path : Path, optional
            Base path of the figures. Nothing is saved if None.
        limit : bool, optional
            Whether to also save the zoomed plot. Default is True.
        figure_format : str, optional
            '.png' or '.svg'. Default is '.png'.
        disp_border : float, optional
            Half-width of the zoomed plot in ms. Default is 60 ms.
        """
        if figure_format not in ('.png', '.svg'):
            raise ValueError('Invalid file format! Must be .png or .svg.')
        save_kwargs = {} if figure_format == '.png' else dict(format='svg', dpi=1200, transparent=True)

        lag_ms = lag / self.fs * 1000
        lag_peak_ms = lag_peak / self.fs * 1000

        plt.figure()
        plt.plot(lag_ms, corr_real, label='real')
        plt.plot(lag_ms, corr_imag, label='imag')
        plt.plot(lag_ms, env, label='envelope', c='k')
        plt.plot(lag_ms, -env, c='k')
        plt.axvline(lag_peak_ms, c='silver', ls='--', label=f'{lag_peak_ms:.2f} ms')
        plt.legend(loc='upper right', ncol=2)
        plt.xlabel('Lag [ms]')
        plt.ylabel('Correlation')
        plt.title('Correlation of DP waveform vs. microphone recording')
        plt.grid()
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path.with_stem(save_path.stem + '_complete').with_suffix(figure_format), **save_kwargs)
        if limit:
            plt.xlim(-disp_border, disp_border)
            plt.ylim(-2 * mag, 2 * mag)
            if save_path is not None:
                plt.savefig(save_path.with_stem(save_path.stem + '_limits').with_suffix(figure_format), **save_kwargs)
        plt.close()

    def plot_DPOAE_MSC(self, result, title, show_corrected=True, save_path=None):
        """Plot an MSC spectrum with the DPOAE band shaded.

        Saves `save_path` (250-3000 Hz) and `<save_path stem>_zoomed.png` (DPOAE band ±100 Hz).

        Parameters
        ----------
        result : dict
            Output of `compute_DPOAE_MSC` (or a grand average with the same keys).
        title : str
            Plot title.
        show_corrected : bool, optional
            Plot the bias-corrected instead of the raw MSC. Default is True.
        save_path : Path, optional
            Path of the figure. Nothing is saved if None.
        """
        fig, ax = plt.subplots()
        freqs = result['freqs']
        y = result['msc_corrected' if show_corrected else 'msc']
        label = 'MSC (bias-corrected)' if show_corrected else 'MSC'
        H0 = 0.0 if show_corrected else result['expected_msc_H0']

        ax.plot(freqs, y, lw=0.9, color='#4C72B0', label=label)
        ax.axhline(H0, color='gray', lw=1, ls='--', label=f"H₀ baseline ({'0' if show_corrected else '1/K'})")

        band_freqs = freqs[result['band_mask']]
        if band_freqs.size:
            ax.axvspan(band_freqs[0], band_freqs[-1], alpha=0.15, color='orange', label='DPOAE band')

        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Coherence')
        ax.set_title(title)
        ax.set_xlim(250, 3000)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            if band_freqs.size:
                ax.set_xlim(np.round(band_freqs[0]) - 100, np.round(band_freqs[-1]) + 100)
                plt.savefig(save_path.with_name(save_path.stem + '_zoomed.png'))
        plt.close()

    # ------------------------------------------------------------------
    # analysis entry points
    # ------------------------------------------------------------------

    def analyze_PT_DPOAE(self, pt_dp=800):
        """Check whether the pure-tone DPOAE at `pt_dp` Hz is significant.

        The peak of the PSD within ±5 % of `pt_dp` is compared with the `sig_border`
        percentile of the PSD 30-70 Hz below and above `pt_dp`.

        Parameters
        ----------
        pt_dp : float, optional
            Frequency of the pure-tone DPOAE in Hz (2*f1 - f2 with f1 = 1 kHz, f2 = 1.2 kHz).
            Default is 800 Hz.

        Returns
        -------
        bool
            Whether the pure-tone DPOAE is significant.
        """
        freq, _, psd_dB = self.compute_PT_PSD()

        pt_indices = np.where((freq >= 0.95 * pt_dp) & (freq <= 1.05 * pt_dp))[0]
        pt_peak = np.max(psd_dB[pt_indices])

        noise_low = np.where((freq > pt_dp - 70) & (freq < pt_dp - 30))[0]
        noise_high = np.where((freq > pt_dp + 30) & (freq < pt_dp + 70))[0]
        noise_level = psd_dB[np.concatenate((noise_low, noise_high))]

        return bool(pt_peak > np.percentile(noise_level, self.sig_border))

    def analyze_speech_DPOAE(self, outofear=False):
        """Evaluate all speech-like DPOAE recordings of the ear contralateral to the audiobook.

        For every trial, the equipment delay is removed, and the recording is evaluated
        against the DP waveforms of the target and the noise voice with the cross-correlation,
        MSC and PSD methods. Results are written to
        `res_path/DPOAE_analysis_contra/<Successful|Unsuccessful>_attention/Target_<voice>/<harm_cond>/<Attended|Ignored>_DPs/`,
        where trials with fewer than two correctly answered comprehension questions are
        sorted into 'Unsuccessful_attention'. Summary tables are written to `res_path`
        (`stats_analysis_DPOAE_contra.csv`, `analysis_volume_contra.csv` and, for in-ear
        recordings, `analysis_CQ.csv`).

        Parameters
        ----------
        outofear : bool, optional
            True for out-of-ear control recordings: no comprehension questions are evaluated
            and all trials count as successful. Default is False.
        """
        res_path = self.res_path / 'DPOAE_analysis_contra'
        res_path.mkdir(parents=True, exist_ok=True)

        columns_cq = ['Harmonic condition', 'Chapter', 'Target', 'Given A1', 'Given A2', 'Given A3', 'Correct A1',
                      'Correct A2', 'Correct A3', 'Number Correct Answers', 'Listening Effort']
        columns_stats = ['Harmonic condition', 'Chapter', 'Target', 'Stimulus Shortcut', 'Number Correct Answers',
                         'ATTENDED Delay [frames]', 'ATTENDED Mag [a.u.]', 'ATTENDED SNR [dB]', 'ATTENDED Significant Peak',
                         'IGNORED Delay [frames]', 'IGNORED Mag [a.u.]', 'IGNORED SNR [dB]', 'IGNORED Significant Peak',
                         'Equipment Delay [frames]', 'Stimulus Path']
        columns_vol = ['Harmonic condition', 'Chapter', 'Target', 'Mean Volume Stimulus [dB SPL]',
                       'Max Volume Stimulus [dB SPL]']

        target_info = self.load_target_info()

        if not outofear:
            exp_info = self.load_experiment_info()
            given_answers = self.get_given_answers(exp_info)
            correct_answers = self.get_correct_answers(target_info)
            count_correct_answers = np.sum(self.compare_answers(given_answers, correct_answers), axis=0)
            listening_effort = self.get_listening_effort(exp_info)
            passing_limit = 2  # trials with fewer correct answers are sorted into 'Unsuccessful_attention'

        rec_paths = self.load_dpoae_recordings()
        stim_paths = target_info['Stimulus'].to_list()
        chapters = target_info['Chapter'].to_list()
        harm_conds = target_info['Harmonic Condition'].to_list()
        targets = target_info['Target'].to_list()
        repetitions = target_info['Repetition'].to_list()

        dpoae_bandwidths = self.load_dpoae_bandwidths()

        print('\nAnalyzing DPOAE recordings from contralateral side ...')

        list_stats, list_vol = [], []
        for i, rep in enumerate(repetitions):
            print(f'Recording {rep} of {len(repetitions)}...')

            target = targets[i].strip()
            stim_path = Path(stim_paths[i])
            rec_path = Path(rec_paths[i])

            # check that the recording matches repetition, chapter, harmonic condition and target
            rec_name_parts = rec_path.name.split('_')
            assert int(rec_name_parts[0]) == rep, f'Recording {rec_path} does not match repetition {rep} in target info!'
            assert int(rec_name_parts[1][2:]) == chapters[i], f'Recording {rec_path} does not match chapter {chapters[i]} in target info!'
            assert rec_name_parts[2:4] == harm_conds[i].split('_'), f'Recording {rec_path} does not match harmonic condition {harm_conds[i]} in target info!'
            assert rec_name_parts[5] == target, f'Recording {rec_path} does not match target {target} in target info!'

            oae_rec = librosa.load(rec_path, sr=self.fs, mono=True)[0]
            stimulus = librosa.load(self.exp_path / stim_path, sr=self.fs, mono=False)[0]

            mean_vol_stim, max_vol_stim = self.estimate_level(oae_rec)

            if outofear:
                sub_path = res_path / 'Successful_attention'
                no_correct_answers = None
            else:
                no_correct_answers = count_correct_answers[i]
                sub_path = res_path / ('Unsuccessful_attention' if no_correct_answers < passing_limit else 'Successful_attention')

            stim_name, wf_dp_attended, wf_dp_ignored = self.load_dp_waveforms(stim_path)

            if target not in ('female', 'male'):
                raise ValueError('Invalid target!')
            save_path = sub_path / f'Target_{target}'
            save_path_attended = save_path / harm_conds[i] / 'Attended_DPs'
            save_path_ignored = save_path / harm_conds[i] / 'Ignored_DPs'
            save_path_attended.mkdir(parents=True, exist_ok=True)
            save_path_ignored.mkdir(parents=True, exist_ok=True)

            # remove the equipment delay from the recording, then check that it is gone
            eq_delay = self.compute_equipment_delay(stimulus, oae_rec, save_path=save_path / f'Ch{chapters[i]}_eq_delay')
            oae_rec = oae_rec[int(eq_delay):]
            eq_delay_check = np.round(self.compute_equipment_delay(stimulus, oae_rec))
            if eq_delay_check > 50:
                raise ValueError(f'Equipment delay not removed correctly! Expected 0, got {eq_delay_check} frames.')
            eq_delay = 0

            # make sure that recording, stimulus and DP waveforms have the same length
            n = min(stimulus.shape[1], len(wf_dp_attended), len(wf_dp_ignored), len(oae_rec))
            wf_dp_attended, wf_dp_ignored, oae_rec = wf_dp_attended[:n], wf_dp_ignored[:n], oae_rec[:n]

            # cross-correlation
            stats_attended = self.compute_DPOAE_crosscorr(wf_dp_attended, oae_rec, eq_delay, save_path_attended / f'Ch{chapters[i]}_')
            stats_ignored = self.compute_DPOAE_crosscorr(wf_dp_ignored, oae_rec, eq_delay, save_path_ignored / f'Ch{chapters[i]}_')

            list_stats.append([harm_conds[i], chapters[i], target, stim_name, no_correct_answers]
                              + stats_attended + stats_ignored + [eq_delay, stim_path])
            list_vol.append([harm_conds[i], chapters[i], target, mean_vol_stim, max_vol_stim])

            # MSC
            dp_band_atd = dpoae_bandwidths[harm_conds[i]][stim_name.split('_')[1]]
            dp_band_ign = dpoae_bandwidths[harm_conds[i]][stim_name.split('_')[3]]

            dict_msc_atd = self.compute_DPOAE_MSC(wf_dp_attended, oae_rec, dp_band_atd)
            dict_msc_ign = self.compute_DPOAE_MSC(wf_dp_ignored, oae_rec, dp_band_ign)
            self.plot_DPOAE_MSC(dict_msc_atd, title=f'{harm_conds[i]} Target {target} Attended DPs',
                                save_path=save_path_attended / f'Ch{chapters[i]}_MSC_plot.png')
            self.plot_DPOAE_MSC(dict_msc_ign, title=f'{harm_conds[i]} Target {target} Ignored DPs',
                                save_path=save_path_ignored / f'Ch{chapters[i]}_MSC_plot.png')
            np.save(save_path_attended / f'Ch{chapters[i]}_MSC_results.npy', dict_msc_atd, allow_pickle=True)
            np.save(save_path_ignored / f'Ch{chapters[i]}_MSC_results.npy', dict_msc_ign, allow_pickle=True)

            # PSD
            dict_psd_atd = self.compute_DPOAE_PSD(f'{harm_conds[i]}_Target{target[0].capitalize()}_attended', oae_rec,
                                                  dp_band_atd, save_path_attended / f'Ch{chapters[i]}_')
            dict_psd_ign = self.compute_DPOAE_PSD(f'{harm_conds[i]}_Target{target[0].capitalize()}_ignored', oae_rec,
                                                  dp_band_ign, save_path_ignored / f'Ch{chapters[i]}_')
            np.save(save_path_attended / f'Ch{chapters[i]}_PSD_results.npy', dict_psd_atd, allow_pickle=True)
            np.save(save_path_ignored / f'Ch{chapters[i]}_PSD_results.npy', dict_psd_ign, allow_pickle=True)

        if not outofear:
            c = np.array(given_answers).shape[1]
            list_cq = np.concatenate((np.array(list_stats)[:, :3], np.array(given_answers).T, np.array(correct_answers).T,
                                      np.array(count_correct_answers).reshape(c, 1),
                                      np.array(listening_effort).reshape(c, 1)), axis=1)
            self.write_results_csv(columns_cq, list_cq, self.res_path / 'analysis_CQ.csv')

        self.write_results_csv(columns_stats, list_stats, self.res_path / 'stats_analysis_DPOAE_contra.csv')
        self.write_results_csv(columns_vol, list_vol, self.res_path / 'analysis_volume_contra.csv')


# The base class implements the JS26 dataset; this alias makes that explicit.
DPOAEProcessorJS26 = DPOAEProcessor
