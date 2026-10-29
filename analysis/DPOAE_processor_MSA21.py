"""Per-participant processing of the MSA21 dataset (Saiz-Alía & Reichenbach, 2021).

`DPOAEProcessorMSA21` evaluates the MSA21 recordings with the methods of `DPOAEProcessor`,
so that results are directly comparable with JS25 and JS26. The folder layout is that of
the original dataset::

    exp_path/
        data/
            <participant>_DONE/
                responses.txt                       answers to the comprehension questions
                section_<k>/
                    results_details.txt             DPOAE voice, DPOAE section, attended yes/no
                    recording_tones.npy             microphone recording
        Waveforms_Female/section_<k>/<n>_F0.mat     harmonic (n = 6, 8) and DP (n = 4) waveforms
        Waveforms_Male/section_<k>/<n>_F0.mat       harmonic (n = 9, 11) and DP (n = 7) waveforms
        results_<participant>/                      created by this class

The out-of-ear control recordings (`analyze_speech_DPOAE_outofear`) were made later with
the JS25 setup and use the layout described in that method.
"""

import glob
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.io import loadmat

from functions_crosscorr import cross_corr
from plot_grand_averages import plot_GA_crosscorr
from DPOAE_processor import DPOAEProcessor


class DPOAEProcessorMSA21(DPOAEProcessor):
    """Evaluate the speech-like DPOAE recordings of one participant of the MSA21 dataset.

    In MSA21, each competing-speaker section contains a DPOAE stimulus derived from either
    the female or the male voice, which was either attended or ignored. Each recording is
    therefore evaluated against a single DP waveform.

    Parameters
    ----------
    subject : str
        Participant ID, e.g. 'P11'. Use 'Autocorrelations' for `compute_autocorrelations`
        and `estimate_DPOAE_freqbands`.
    exp_path : str or Path
        Parent folder of the dataset (see the module docstring for its layout).
    res_path : str or Path, optional
        Folder the results are written to. Default is `exp_path/results_<subject>`.

    Attributes
    ----------
    ndp : dict
        Harmonic numbers of the DP waveform ('dp') and of the two stimulus components
        ('stimuli') for the female and male voice.

    All other attributes are as in `DPOAEProcessor`; `gain` is 0 dB for this dataset.
    """

    def __init__(self, subject, exp_path, res_path=None):
        self.subject = subject
        self.exp_path = Path(exp_path)
        self.data_path = self.exp_path / 'data'
        self.res_path = self.exp_path / f'results_{subject}' if res_path is None else Path(res_path)
        self.res_path.mkdir(parents=True, exist_ok=True)

        self.sig_border = 97  # percentile for significance
        self.pref = 20e-6  # reference pressure in Pa

        self.fs = 44100  # sampling frequency
        self.gain = 0  # no microphone gain is subtracted for this dataset
        self.sens_mV = 50e-3  # microphone sensitivity in V/Pa

        self.ndp = {'Female': {'dp': 4, 'stimuli': [6, 8]},
                    'Male': {'dp': 7, 'stimuli': [9, 11]}}

    def analyze_PT_DPOAE(self, pt_dp=800):
        """Not available: the MSA21 dataset contains no pure-tone DPOAE recordings."""
        raise NotImplementedError('The MSA21 dataset contains no pure-tone DPOAE recordings.')

    def _load_waveform(self, mat_path):
        """Load a waveform stored as 'shifted_F0' in a .mat file and return it as a 1-D array."""
        return np.squeeze(loadmat(mat_path)['shifted_F0'])

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
        return np.load(bandwidths_path, allow_pickle=True).item()

    def estimate_DPOAE_freqbands(self):
        """Estimate the frequency band of the DPOAE for the female and male voice with the
        same method as for JS26 (see `DPOAEProcessor.estimate_DPOAE_freqbands`). The result
        is saved to `exp_path/dpoae_bandwidths.npy`.

        Returns
        -------
        dict
            {'Female': (f_low, f_high), 'Male': (f_low, f_high)} in Hz.
        """
        dpoae_bandwidths = {}
        for voice, numbers in self.ndp.items():
            all_dp_files = sorted((self.exp_path / f'Waveforms_{voice}').rglob(f"{numbers['dp']}_F0.mat"))
            results_low, results_high = [], []
            for file in all_dp_files:
                _, _, f_low, f_high = self.estimate_stim_frequency(self._load_waveform(file), return_median=False,
                                                                   return_min_max=True)
                results_low.append(f_low)
                results_high.append(f_high)
            dpoae_bandwidths[voice] = (np.mean(results_low), np.mean(results_high))

        np.save(self.exp_path / 'dpoae_bandwidths.npy', dpoae_bandwidths, allow_pickle=True)
        return dpoae_bandwidths

    def compute_autocorrelations(self):
        """Compute the autocorrelation of every DP waveform and their average per voice
        (see `DPOAEProcessor.compute_autocorrelations`).

        The averages are saved in `res_path/GA_autocorr_dict.npz` with the keys 'F Attended',
        'F Ignored', 'M Attended' and 'M Ignored'; attended and ignored share the same
        waveforms in this dataset.
        """
        GA_dict = {}
        for voice, numbers in self.ndp.items():
            all_dp_files = sorted((self.exp_path / f'Waveforms_{voice}').rglob(f"{numbers['dp']}_F0.mat"))
            save_path = self.res_path / f'{voice}_DPs'
            save_path.mkdir(parents=True, exist_ok=True)

            auto_ga = np.array([])
            for file in all_dp_files:
                dp_stim = self._load_waveform(file)
                auto_lag, auto_real, auto_imag, auto_env = cross_corr(dp_stim, dp_stim)
                auto_stats = np.array([auto_lag, auto_real, auto_imag, auto_env])
                auto_ga = np.append(auto_ga, auto_stats)
                np.save(save_path / f'autocorr_{Path(file).parent.name}.npy', auto_stats)

                lag_peak, mag_peak, _, _ = self.find_significant_peak(auto_lag, auto_env)
                self.plot_DPOAE_crosscorr(auto_lag, auto_real, auto_imag, auto_env, lag_peak, mag_peak,
                                          save_path / f'autocorr_{Path(file).parent.name}.png')

            auto_ga_result = np.mean(auto_ga.reshape(len(all_dp_files), 4, -1), axis=0)
            plot_GA_crosscorr(f'Autocorrelation {voice}', auto_ga_result, save_path / f'GA_autocorr_{voice}.png')
            np.save(save_path / f'GA_autocorr_{voice}.npy', auto_ga_result)
            GA_dict[f'{voice[0]} Attended'] = auto_ga_result
            GA_dict[f'{voice[0]} Ignored'] = auto_ga_result

        np.savez_compressed(self.res_path / 'GA_autocorr_dict.npz', **GA_dict)

    def _evaluate_recording(self, oae_rec, stimuli, wf_dp, voice, band, save_path, label, eq_plot_path):
        """Remove the equipment delay from a recording and evaluate it against one DP waveform.

        Parameters
        ----------
        oae_rec : np.ndarray
            Microphone recording.
        stimuli : np.ndarray
            The two stimulus components, shape (2, n).
        wf_dp : np.ndarray
            Expected DP waveform.
        voice : str
            'Female' or 'Male'; selects the envelope smoothing (see below).
        band : sequence of float
            DPOAE frequency band in Hz.
        save_path : Path
            Folder of the condition the results are written to.
        label : str
            Prefix of the output files, e.g. 'Sec3'.
        eq_plot_path : Path
            Path of the equipment-delay diagnostic plot.

        Returns
        -------
        list
            Cross-correlation statistics as returned by `compute_DPOAE_crosscorr`.
        """
        eq_delay = self.compute_equipment_delay([stimuli[0], stimuli[1]], oae_rec, eq_plot_path)
        oae_rec = oae_rec[int(eq_delay):]
        eq_delay_check = np.round(self.compute_equipment_delay([stimuli[0], stimuli[1]], oae_rec))
        if eq_delay_check > 50:
            raise ValueError(f'Equipment delay not removed correctly for {label}! Expected ~0, got {eq_delay_check} frames.')

        n = min(stimuli.shape[1], len(wf_dp), len(oae_rec))
        wf_dp, oae_rec = wf_dp[:n], oae_rec[:n]

        # voice-specific envelope smoothing for the MSA21 DP waveforms
        smooth_env_cf = 2 if voice == 'Female' else 0.1
        stats = self.compute_DPOAE_crosscorr(wf_dp, oae_rec, 0, save_path / f'{label}_', smooth_env_cutoff=smooth_env_cf)

        dict_msc = self.compute_DPOAE_MSC(wf_dp, oae_rec, band)
        self.plot_DPOAE_MSC(dict_msc, title=f'{label} {save_path.name}', save_path=save_path / f'{label}_MSC_plot.png')
        np.save(save_path / f'{label}_MSC_results.npy', dict_msc, allow_pickle=True)

        dict_psd = self.compute_DPOAE_PSD(f'{label}_{save_path.name}', oae_rec, band, save_path / f'{label}_')
        np.save(save_path / f'{label}_PSD_results.npy', dict_psd, allow_pickle=True)

        return stats

    def analyze_speech_DPOAE(self):
        """Evaluate the competing-speaker sections (3 to 14) of one participant.

        Each recording is evaluated against the DP waveform of its section with the
        cross-correlation, MSC and PSD methods of `DPOAEProcessor`. Results are written to
        `res_path/DPOAE_analysis/Successful_attention/<Attended|Ignored>_<female|male>/`.
        The answers to the comprehension questions are only checked for completeness: no
        answer key is available, so every section counts as successful. Summary tables are
        written to `res_path` (`stats_analysis_DPOAE.csv`, `analysis_volume.csv`).
        """
        res_path = self.res_path / 'DPOAE_analysis'
        res_path.mkdir(parents=True, exist_ok=True)

        columns_stats = ['Rep', 'Section', 'DP Chapter', 'Attended', 'Condition', 'DP Stimulus',
                         'DP Delay [frames]', 'DP Mag [a.u.]', 'DP SNR [dB]', 'DP Significant Peak', 'Equipment Delay [frames]']
        columns_vol = ['Section', 'Condition', 'Mean Volume Stimulus [dB SPL]', 'Max Volume Stimulus [dB SPL]']

        with open(self.data_path / f'{self.subject}_DONE/responses.txt') as f:
            values = [v for v in f.read().strip().split('|') if v != '']
        given_answers = values[1::2][2:]  # drop the answers of the single-speaker sections
        if len(given_answers) != 12:
            raise ValueError(f'Number of answered comprehension questions is {len(given_answers)} instead of 12!')

        dpoae_bandwidths = self.load_dpoae_bandwidths()
        conditions = {('Male', 'Yes'): 'Attended_male', ('Male', 'No'): 'Ignored_male',
                      ('Female', 'Yes'): 'Attended_female', ('Female', 'No'): 'Ignored_female'}

        print('\nAnalyzing DPOAE recordings...')

        list_stats, list_vol = [], []
        for i, sec in enumerate(range(3, 15)):
            print(f'Recording number {i + 1}...')

            section_path = self.data_path / f'{self.subject}_DONE/section_{sec}/'
            with open(section_path / 'results_details.txt') as rd:
                results_details = dict(part.strip().split(': ') for part in rd.read().split(','))
            dp_voice = results_details['DPOAE gender']
            dp_chapter = results_details['DPOAE section']
            condition = conditions[(dp_voice, results_details['Attended'])]

            oae_rec = np.squeeze(np.load(section_path / 'recording_tones.npy'))
            wf_path = self.exp_path / f'Waveforms_{dp_voice}/section_{dp_chapter}'
            wf_dp = self._load_waveform(wf_path / f"{self.ndp[dp_voice]['dp']}_F0.mat")
            stimuli = np.array([self._load_waveform(wf_path / f'{n}_F0.mat') for n in self.ndp[dp_voice]['stimuli']])

            mean_vol_stim, max_vol_stim = self.estimate_level(oae_rec)

            save_path = res_path / 'Successful_attention' / condition
            save_path.mkdir(parents=True, exist_ok=True)

            stats = self._evaluate_recording(oae_rec, stimuli, wf_dp, dp_voice, dpoae_bandwidths[dp_voice], save_path,
                                             f'Sec{sec}', save_path.parent / f'Sec{sec}_eq_delay')

            list_stats.append([i + 1, sec, dp_chapter, results_details['Attended'], condition, dp_voice] + stats + [0])
            list_vol.append([sec, condition, mean_vol_stim, max_vol_stim])

        self.write_results_csv(columns_stats, list_stats, self.res_path / 'stats_analysis_DPOAE.csv')
        self.write_results_csv(columns_vol, list_vol, self.res_path / 'analysis_volume.csv')

    def analyze_speech_DPOAE_outofear(self):
        """Evaluate the out-of-ear control recordings made with the MSA21 stimuli.

        These recordings were made with the JS25 setup and use a different layout than the
        participant data: the trial order is read from `data/<subject>_target_info.csv`
        (columns 'Chapter', 'Target', e.g. 'Female Att', and 'Stimulus folder', relative to
        `exp_path`), and the recordings from
        `data/<subject>_*_mic_Contralateral_recorded/recording_mic_Contralateral_*.wav`.
        Results are written in the same structure as by `analyze_speech_DPOAE`.
        """
        res_path = self.res_path / 'DPOAE_analysis'
        res_path.mkdir(parents=True, exist_ok=True)

        columns_stats = ['Rep', 'DP Chapter', 'Condition', 'DP Stimulus', 'DP Delay [frames]', 'DP Mag [a.u.]',
                         'DP SNR [dB]', 'DP Significant Peak', 'Equipment Delay [frames]']

        target_info = pd.read_csv(self.data_path / f'{self.subject}_target_info.csv', sep='\t')
        chapters = target_info['Chapter'].values
        targets = target_info['Target'].values
        stim_folders = target_info['Stimulus folder'].values

        oae_recs = sorted(glob.glob(str(self.data_path / f'{self.subject}_*_mic_Contralateral_recorded/recording_mic_Contralateral_*.wav')))
        assert len(oae_recs) == len(chapters) == len(targets) == len(stim_folders), \
            'Mismatch in number of recordings and target info entries!'

        dpoae_bandwidths = self.load_dpoae_bandwidths()

        print('\nAnalyzing DPOAE recordings (outofear)...')

        list_stats = []
        for i, stim_folder in enumerate(stim_folders):
            print(f'Recording number {i + 1}...')

            voice = targets[i].split()[0]
            oae_rec = np.squeeze(librosa.load(oae_recs[i], sr=self.fs)[0])
            wf_path = self.exp_path / stim_folder
            wf_dp = self._load_waveform(wf_path / f"{self.ndp[voice]['dp']}_F0.mat")
            stimuli = np.array([self._load_waveform(wf_path / f'{n}_F0.mat') for n in self.ndp[voice]['stimuli']])

            # e.g. 'Female Att' -> 'Attended_female'
            target = targets[i].lower().replace('ign', 'Ignored').replace('att', 'Attended')
            condition = '_'.join(target.split()[::-1])

            save_path = res_path / 'Successful_attention' / condition
            save_path.mkdir(parents=True, exist_ok=True)

            stats = self._evaluate_recording(oae_rec, stimuli, wf_dp, voice, dpoae_bandwidths[voice], save_path,
                                             f'Rep{i + 1}', save_path.parent / f'Rep{i + 1}_eq_delay')

            list_stats.append([i + 1, chapters[i], condition, voice] + stats + [0])

        self.write_results_csv(columns_stats, list_stats, self.res_path / 'stats_analysis_DPOAE.csv')
