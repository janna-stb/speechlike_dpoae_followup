"""Per-participant processing of the JS25 dataset (Steinebach & Reichenbach, 2026).

`DPOAEProcessorJS25` evaluates the JS25 recordings with the methods of `DPOAEProcessor`, so
that results are directly comparable with JS26 and MSA21. Only the data-loading methods
that depend on the JS25 file layout are overridden.

The expected folder layout is that of the published dataset (Zenodo, doi:10.5281/zenodo.16837673)::

    exp_path/
        Recordings/
            data_<participant>/
                <participant>_<timestamp>.xlsx                      PsychoPy experiment data
                <participant>_targets_Attention.csv                 order of attentional targets
                <participant>_targets_Singlespeaker.csv             single-speaker order
                <participant>_<timestamp>_mic_stim_Attention_recorded/recording_mic_*.wav
                <participant>_<timestamp>_mic_stim_SingleSpeaker_recorded/recording_mic_*.wav
        Stimuli/
            <Book>_<voice>/dp_n<k>_<Book>_<voice>_<chapter>.wav     DP waveforms
            ...                                                     presented stimuli (paths in the target csv files)
        results_<participant>/                                      created by this class
"""

import glob
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from functions_crosscorr import cross_corr
from plot_grand_averages import plot_GA_crosscorr
from DPOAE_processor import DPOAEProcessor

# np.trapz was renamed to np.trapezoid in NumPy 2.0 (and later removed)
trapezoid = getattr(np, 'trapezoid', None) or np.trapz


class DPOAEProcessorJS25(DPOAEProcessor):
    """Evaluate the speech-like DPOAE recordings of one participant of the JS25 dataset.

    In JS25, each stimulus contains a resolved and an unresolved harmonic pair, so every
    recording is evaluated against two DP waveforms per voice. The attended voice is always
    read by the speaker of the book 'Polarnacht', the ignored voice by the speaker of 'Darum'.
    In chapters with target 'video', participants watched a silent video and ignored both
    voices (baseline condition).

    Parameters
    ----------
    subject : str
        Participant ID, e.g. 'P01'. Use 'Autocorrelations' for `compute_autocorrelations`
        and `estimate_DPOAE_freqbands`.
    exp_path : str or Path
        Parent folder of the dataset (see the module docstring for its layout).
    res_path : str or Path, optional
        Folder the results are written to. Default is `exp_path/results_<subject>`.

    Attributes
    ----------
    harm_number_map : dict
        Harmonic number of the DP waveform (as in its file name) for the resolved and
        unresolved component of each book and voice.

    All other attributes are as in `DPOAEProcessor`.
    """

    def __init__(self, subject, exp_path, res_path=None):
        # JS25 has a different harmonic-condition structure than JS26, so the shared
        # attributes are set up here instead of calling DPOAEProcessor.__init__
        self.subject = subject
        self.exp_path = Path(exp_path)
        self.data_path = self.exp_path / 'Recordings' / f'data_{subject}'
        self.stim_path = self.exp_path / 'Stimuli'
        self.res_path = self.exp_path / f'results_{subject}' if res_path is None else Path(res_path)
        self.res_path.mkdir(parents=True, exist_ok=True)

        self.sig_border = 97  # percentile for significance
        self.pref = 20e-6  # reference pressure in Pa

        self.fs = 44100  # sampling frequency
        self.gain = 40  # microphone gain of 40 dB
        self.sens_mV = 50e-3  # microphone sensitivity in V/Pa

        # harmonic number of the DP waveform: female resolved n7/n9 -> 5, male resolved n6/n8 -> 4,
        # unresolved n15/n18 -> 12 for both voices
        n_F_res, n_M_res, n_unres = 5, 4, 12
        self.harm_number_map = {
            'Polarnacht_female': {'resolved': n_F_res, 'unresolved': n_unres},
            'Polarnacht_male': {'resolved': n_M_res, 'unresolved': n_unres},
            'Darum_female': {'resolved': n_F_res, 'unresolved': n_unres},
            'Darum_male': {'resolved': n_M_res, 'unresolved': n_unres},
        }

    # ------------------------------------------------------------------
    # experiment and target information
    # ------------------------------------------------------------------

    def clean_header(self, columns):
        """Replace unnamed columns of an excel sheet with the name of the preceding column.

        Parameters
        ----------
        columns : list of str
            Column names as read by pandas.

        Returns
        -------
        list of str
            The cleaned column names.
        """
        cleaned_header = []
        current_header = None
        for col in columns:
            if 'Unnamed' not in col:
                current_header = col
            cleaned_header.append(current_header)
        return cleaned_header

    def load_experiment_info(self, speaker_type, loop_reps=6):
        """Load the experiment information from the excel file written by PsychoPy.

        Parameters
        ----------
        speaker_type : str
            'Single' or 'Competing'.
        loop_reps : int, optional
            Number of repetitions of the competing-speaker loop (one sheet each). Default is 6.

        Returns
        -------
        excel_info : pd.DataFrame
            One row per loop repetition.
        """
        excel_file = glob.glob(str(self.data_path / f'{self.subject}_*.xlsx'))
        if len(excel_file) != 1:
            raise ValueError(f'Found {len(excel_file)} excel files for {self.subject}!')
        excel_path = excel_file[0]

        if speaker_type == 'Single':
            excel_info = pd.read_excel(excel_path, sheet_name='loop_SingleSpeaker').iloc[:1, :]
        elif speaker_type == 'Competing':
            sheets = ['loop_Attention'] + [f'loop_Attention{i}' for i in range(1, loop_reps)]
            excel_info = pd.concat([pd.read_excel(excel_path, sheet_name=s).iloc[:1, :] for s in sheets],
                                   ignore_index=True)
        else:
            raise ValueError(f"Unknown speaker type: {speaker_type}. Use 'Single' or 'Competing'.")

        excel_info.columns = self.clean_header(excel_info.columns)
        return excel_info

    def load_target_info(self, speaker_type):
        """Load the trial order (targets, chapters, stimulus paths, correct answers).

        Parameters
        ----------
        speaker_type : str
            'Single' or 'Competing'.

        Returns
        -------
        target_info : pd.DataFrame
            Content of `<subject>_targets_Singlespeaker.csv` or `<subject>_targets_Attention.csv`.
        """
        if speaker_type == 'Single':
            return pd.read_csv(self.data_path / f'{self.subject}_targets_Singlespeaker.csv', sep='\t')
        if speaker_type == 'Competing':
            return pd.read_csv(self.data_path / f'{self.subject}_targets_Attention.csv', sep='\t')
        raise ValueError(f"Unknown speaker type: {speaker_type}. Use 'Single' or 'Competing'.")

    def get_given_answers(self, excel_info):
        """Extract the given answers to the three comprehension questions.

        Answers were given with a slider in most sessions and with key presses in some; both
        are supported.

        Parameters
        ----------
        excel_info : pd.DataFrame
            Experiment information as returned by `load_experiment_info`.

        Returns
        -------
        answers : list of list of str
            The given answers ('a', 'b' or 'c') in the form [[Q1], [Q2], [Q3]].
        """
        answers = []
        for i in range(1, 4):
            sub_answers = []
            try:
                for j in range(excel_info.loc[:, f'slider_Q{i}.response_raw'].shape[0]):
                    answer = excel_info.loc[j, f'slider_Q{i}.response_raw'].tolist()
                    sub_answers.extend(['c' if item == 1 else 'b' if item == 2 else 'a' for item in answer])
            except Exception:
                sub_answers = []
                for j in range(excel_info.loc[:, f'key_Q{i}.keys_raw'].shape[0]):
                    answer = excel_info.loc[j, f'key_Q{i}.keys_raw'].tolist()
                    sub_answers.extend([item.strip().strip("'") for item in answer])
            answers.append(sub_answers)
        return answers

    def get_listening_effort(self, excel_info):
        """Extract the listening-effort ratings.

        Parameters
        ----------
        excel_info : pd.DataFrame
            Experiment information as returned by `load_experiment_info`.

        Returns
        -------
        list
            One rating per chapter.
        """
        return excel_info.loc[:, 'slider.response_raw'].to_numpy().flatten().tolist()

    # ------------------------------------------------------------------
    # recordings and waveforms
    # ------------------------------------------------------------------

    def analyze_PT_DPOAE(self, pt_dp=800):
        """Not available: the JS25 dataset contains no pure-tone DPOAE recordings."""
        raise NotImplementedError('The JS25 dataset contains no pure-tone DPOAE recordings.')

    def get_recordings(self, speaker_type):
        """Load the microphone recordings of all chapters, in chapter order, with the mean removed.

        Parameters
        ----------
        speaker_type : str
            'Single' or 'Competing'.

        Returns
        -------
        oae_recs : list of np.ndarray
            One recording per chapter.
        """
        if speaker_type == 'Single':
            folder = f'{self.subject}_*_mic_stim_SingleSpeaker_recorded'
        elif speaker_type == 'Competing':
            folder = f'{self.subject}_*_mic_stim_Attention_recorded'
        else:
            raise ValueError("Invalid speaker type! Can either be 'Single' or 'Competing'.")

        oae_recs = []
        for rec_path in sorted(glob.glob(str(self.data_path / folder / 'recording_mic_*.wav'))):
            orec = librosa.load(rec_path, sr=self.fs, mono=True)[0]
            oae_recs.append(orec - np.mean(orec))
        return oae_recs

    def load_dp_waveforms(self, target, chapter, speaker_type, stim, return_dp_path=False):
        """Load the expected DP waveforms (resolved and unresolved) of the attended
        (Polarnacht) and the ignored (Darum) voice for one chapter.

        Parameters
        ----------
        target : str
            Target of the chapter: 'female', 'male' or 'video'.
        chapter : int
            Chapter number.
        speaker_type : str
            'Single' or 'Competing'.
        stim : str
            Path of the presented stimulus; for target 'video' it determines which voice
            pairing was presented.
        return_dp_path : bool, optional
            If True, return the paths of the DP waveforms instead of the waveforms. Default is False.

        Returns
        -------
        stim_name : str
            Pairing of book and voice, e.g. 'Stimuli_Polarnacht_female_Darum_male'.
        wf_dp_attended : np.ndarray or list of str
            DP waveforms of the attended voice, stacked as [resolved, unresolved] (or their paths).
        wf_dp_ignored : np.ndarray or list of str
            DP waveforms of the ignored voice, stacked as [resolved, unresolved] (or their paths).
        """
        chapter = int(chapter)

        def dp_paths(book_voice, ch_no, single=False):
            book, voice = book_voice.split('_')
            suffix = f'single_{ch_no}' if single else ch_no
            codes = self.harm_number_map[book_voice]
            return [f'Stimuli/{book_voice}/dp_n{codes[h]}_{book}_{voice}_{suffix}.wav' for h in ('resolved', 'unresolved')]

        if speaker_type == 'Single':
            if chapter > 2:
                raise ValueError('Chapter > 2 not supported for Single Speaker!')
            ch_no = f'0{chapter}'
            if target == 'female':
                # Darum is not presented in single-speaker trials; its waveforms serve as a control
                dp_path_atd, dp_path_ign = dp_paths('Polarnacht_female', ch_no, single=True), dp_paths('Darum_male', ch_no)
                stim_name = 'Polarnacht_female'
            elif target == 'male':
                dp_path_atd, dp_path_ign = dp_paths('Polarnacht_male', ch_no, single=True), dp_paths('Darum_female', ch_no)
                stim_name = 'Polarnacht_male'
            else:
                raise ValueError('Could not determine attended and ignored speaker for Single Speaker!')

        elif speaker_type == 'Competing':
            ch_no = f'{chapter:02d}'
            if target == 'female' or (target == 'video' and 'MixedTwobandStimuli_Polarnacht_female_Darum_male' in stim):
                pairing = ('Polarnacht_female', 'Darum_male')
            elif target == 'male' or (target == 'video' and 'MixedTwobandStimuli_Darum_female_Polarnacht_male' in stim):
                pairing = ('Polarnacht_male', 'Darum_female')
            elif target == 'video':
                raise ValueError(f"Could not determine DP waveforms for target 'video' from stimulus path: {stim}")
            else:
                raise ValueError('Could not determine attended and ignored speaker for Competing Speaker!')
            dp_path_atd, dp_path_ign = dp_paths(pairing[0], ch_no), dp_paths(pairing[1], ch_no)
            stim_name = f'Stimuli_{pairing[0]}_{pairing[1]}'
        else:
            raise ValueError("Invalid speaker type! Can either be 'Single' or 'Competing'.")

        if return_dp_path:
            return stim_name, dp_path_atd, dp_path_ign

        wf_dp_attended = np.vstack([librosa.load(self.exp_path / p, sr=self.fs, mono=True)[0] for p in dp_path_atd])
        wf_dp_ignored = np.vstack([librosa.load(self.exp_path / p, sr=self.fs, mono=True)[0] for p in dp_path_ign])
        return stim_name, wf_dp_attended, wf_dp_ignored

    def _find_dp_waveform_files(self, book_voice, harm_type):
        """List all DP waveform files (single- and competing-speaker chapters) of one book,
        voice and harmonic type.

        Parameters
        ----------
        book_voice : str
            Key of `harm_number_map`, e.g. 'Polarnacht_female'.
        harm_type : str
            'resolved' or 'unresolved'.

        Returns
        -------
        list of str
            Sorted paths of the DP waveform files.
        """
        book, voice = book_voice.split('_')
        code = self.harm_number_map[book_voice][harm_type]
        folder = self.stim_path / book_voice
        all_dp_files = sorted(glob.glob(str(folder / f'dp_n{code}_{book}_{voice}*.wav')))
        if not all_dp_files:
            raise ValueError(f'No DP waveform files found for {book_voice} ({harm_type}) in {folder}.')
        return all_dp_files

    def estimate_DPOAE_freqbands(self):
        """Estimate the frequency band of the DPOAE for each voice and harmonic type with the
        same method as for JS26 (see `DPOAEProcessor.estimate_DPOAE_freqbands`), averaged
        across books. The result is saved to `exp_path/dpoae_bandwidths.npy`.

        Returns
        -------
        dict
            {'female': {'resolved': (f_low, f_high), 'unresolved': (...)}, 'male': {...}} in Hz.
        """
        bands_per_voice = {v: {h: [] for h in ('resolved', 'unresolved')} for v in ('male', 'female')}

        for book_voice, harm_codes in self.harm_number_map.items():
            voice = book_voice.split('_')[-1]
            for harm_type in harm_codes:
                results_low, results_high = [], []
                for file in self._find_dp_waveform_files(book_voice, harm_type):
                    dp_wf, _ = librosa.load(file, sr=self.fs, mono=True)
                    _, _, f_low, f_high = self.estimate_stim_frequency(dp_wf, return_median=False, return_min_max=True)
                    results_low.append(f_low)
                    results_high.append(f_high)
                bands_per_voice[voice][harm_type].append([np.mean(results_low), np.mean(results_high)])

        dpoae_bandwidths = {voice: {h: (np.mean([bw[0] for bw in bws]), np.mean([bw[1] for bw in bws]))
                                    for h, bws in bws_dict.items()}
                            for voice, bws_dict in bands_per_voice.items()}

        np.save(self.exp_path / 'dpoae_bandwidths.npy', dpoae_bandwidths, allow_pickle=True)
        return dpoae_bandwidths

    def compute_autocorrelations(self):
        """Compute the autocorrelation of every DP waveform and their average per book,
        voice and harmonic type (see `DPOAEProcessor.compute_autocorrelations`).

        All averages are saved together in `res_path/GA_autocorr_dict.npz` with keys such as
        'Resolved F Attended' (Polarnacht = attended, Darum = ignored).
        """
        for book_voice, harm_codes in self.harm_number_map.items():
            for harm_type in harm_codes:
                all_dp_files = self._find_dp_waveform_files(book_voice, harm_type)
                save_path = self.res_path / book_voice / harm_type
                save_path.mkdir(parents=True, exist_ok=True)

                auto_ga = np.array([])
                for file in all_dp_files:
                    dp_stim, _ = librosa.load(file, sr=self.fs, mono=True)
                    auto_lag, auto_real, auto_imag, auto_env = cross_corr(dp_stim, dp_stim)
                    auto_stats = np.array([auto_lag, auto_real, auto_imag, auto_env])
                    auto_ga = np.append(auto_ga, auto_stats)
                    np.save(save_path / f'autocorr_{Path(file).stem}.npy', auto_stats)

                    lag_peak, mag_peak, _, _ = self.find_significant_peak(auto_lag, auto_env)
                    self.plot_DPOAE_crosscorr(auto_lag, auto_real, auto_imag, auto_env, lag_peak, mag_peak,
                                              save_path / f'autocorr_{Path(file).stem}.png')

                auto_ga_result = np.mean(auto_ga.reshape(len(all_dp_files), 4, -1), axis=0)
                plot_GA_crosscorr(f'Autocorrelation {book_voice} {harm_type}', auto_ga_result,
                                  save_path / f'GA_autocorr_{harm_type}.png')
                np.save(save_path / f'GA_autocorr_{harm_type}.npy', auto_ga_result)

        all_GA_files = glob.glob(str(self.res_path / '**/GA_autocorr_*.npy'), recursive=True)
        if len(all_GA_files) > 8:
            raise ValueError(f'Found {len(all_GA_files)} GA files, but expected at most 8!')

        GA_dict = {}
        for file in all_GA_files:
            parts = Path(file).parts
            harm_type, book_voice = parts[-2], parts[-3]
            voice = book_voice.split('_')[-1].capitalize()
            attention = 'Attended' if book_voice.startswith('Polarnacht') else 'Ignored'
            GA_dict[f'{harm_type.capitalize()} {voice[0]} {attention}'] = np.load(file)

        np.savez_compressed(self.res_path / 'GA_autocorr_dict.npz', **GA_dict)

    # ------------------------------------------------------------------
    # baseline (video) grand averages
    # ------------------------------------------------------------------

    def _combine_MSC_dicts(self, dicts):
        """Average two or more MSC grand averages (as saved by `average_MSC_per_condition`).

        The band summaries are recomputed from the averaged spectra. The inputs must share
        the same frequency grid; the combined band mask is the union of the input masks.

        Parameters
        ----------
        dicts : list of dict
            MSC grand averages.

        Returns
        -------
        dict
            Combined grand average with the same keys.
        """
        freqs = dicts[0]['freqs']
        if any(not np.array_equal(d['freqs'], freqs) for d in dicts[1:]):
            raise ValueError('Cannot average MSC dicts on different frequency grids (different nperseg values).')

        band_mask = np.logical_or.reduce([d['band_mask'] for d in dicts])
        msc = np.mean([d['msc'] for d in dicts], axis=0)
        msc_corrected = np.mean([d['msc_corrected'] for d in dicts], axis=0)

        return {'freqs': freqs, 'msc': msc, 'msc_corrected': msc_corrected,
                'msc_band_mean': msc[band_mask].mean(), 'msc_band_max': msc[band_mask].max(),
                'msc_band_mean_corrected': msc_corrected[band_mask].mean(),
                'msc_band_max_corrected': msc_corrected[band_mask].max(),
                'n_segments': dicts[0]['n_segments'], 'band_mask': band_mask, 'nperseg': dicts[0]['nperseg'],
                'df': dicts[0]['df']}

    def _combine_PSD_dicts(self, dicts):
        """Average two or more PSD grand averages (as saved by `average_PSD_per_condition`).

        The band power is recomputed from the averaged PSD. The inputs must share the same
        frequency grid (the Welch segment length depends on the recording length); the
        combined band mask is the union of the input masks.

        Parameters
        ----------
        dicts : list of dict
            PSD grand averages.

        Returns
        -------
        dict
            Combined grand average with the same keys.
        """
        freqs = dicts[0]['freqs']
        if any(not np.array_equal(d['freqs'], freqs) for d in dicts[1:]):
            raise ValueError('Cannot average PSD dicts on different frequency grids (recordings of different length).')

        band_mask = np.logical_or.reduce([d['band_mask'] for d in dicts])
        psd_Pa2 = np.mean([d['psd_Pa2_per_Hz'] for d in dicts], axis=0)
        psd_dB = np.mean([d['psd_dB'] for d in dicts], axis=0)
        band_power_Pa2 = trapezoid(psd_Pa2[band_mask], freqs[band_mask])

        return {'band_power_Pa2': band_power_Pa2,
                'band_power_dB': 10 * np.log10(band_power_Pa2 / self.pref ** 2) - self.gain,
                'freqs': freqs, 'psd_Pa2_per_Hz': psd_Pa2, 'psd_dB': psd_dB, 'band_mask': band_mask,
                'df_Hz': dicts[0]['df_Hz'], 'n_averages': dicts[0]['n_averages']}

    def average_baseline_GAs(self):
        """Combine the baseline (target 'video') grand averages into one per voice.

        In baseline chapters, a voice's DPOAE ends up in two folders, depending on the voice
        pairing that was presented: e.g. the female voice is stored under
        'Target_video_BL_Polarnacht_female/<harm>/Attended_DPs' when Polarnacht was read by the
        female speaker, and under 'Target_video_BL_Polarnacht_male/<harm>/Ignored_DPs' when
        Darum was. This method averages both halves per voice, harmonic type and metric, saves
        the result directly under 'Target_video_BL_Polarnacht_<voice>/<harm>/', and deletes the
        two source files. Groups that have already been combined are skipped, so it is safe to
        call the method more than once.

        Run it after `average_crosscorr_per_condition`, `average_MSC_per_condition` and
        `average_PSD_per_condition`.
        """
        pairsF = [('Polarnacht_female', 'Attended_DPs'), ('Polarnacht_male', 'Ignored_DPs')]
        pairsM = [('Polarnacht_male', 'Attended_DPs'), ('Polarnacht_female', 'Ignored_DPs')]

        bl_path = self.res_path / 'Competing_Speaker' / 'Successful_attention'
        if not bl_path.exists():
            return

        for harm in ['resolved', 'unresolved']:
            for metric in ['crosscorr', 'msc', 'psd']:
                metric_key = f'{metric.upper()}_' if metric != 'crosscorr' else ''
                ga_data = sorted(bl_path.rglob(f'Target_video_BL_*/{harm}/**/GA_{metric_key}data.npy'))

                def _find(keywords):
                    return next((p for p in ga_data if all(kw in str(p) for kw in keywords)), None)

                pathsF = [_find(pairsF[0]), _find(pairsF[1])]
                pathsM = [_find(pairsM[0]), _find(pairsM[1])]
                if any(p is None for p in pathsF + pathsM):
                    continue  # nothing to combine yet, or already combined

                if metric == 'crosscorr':
                    ga_F = np.mean([np.load(p) for p in pathsF], axis=0)
                    ga_M = np.mean([np.load(p) for p in pathsM], axis=0)
                else:
                    combine = self._combine_MSC_dicts if metric == 'msc' else self._combine_PSD_dicts
                    ga_F = combine([np.load(p, allow_pickle=True).item() for p in pathsF])
                    ga_M = combine([np.load(p, allow_pickle=True).item() for p in pathsM])

                np.save(bl_path / 'Target_video_BL_Polarnacht_female' / harm / f'GA_{metric_key}data.npy', ga_F,
                        allow_pickle=(metric != 'crosscorr'))
                np.save(bl_path / 'Target_video_BL_Polarnacht_male' / harm / f'GA_{metric_key}data.npy', ga_M,
                        allow_pickle=(metric != 'crosscorr'))
                for old_file in pathsF + pathsM:
                    old_file.unlink()

    # ------------------------------------------------------------------
    # analysis entry point
    # ------------------------------------------------------------------

    def analyze_speech_DPOAE(self, speaker_type, outofear=False):
        """Evaluate all speech-like DPOAE recordings of one speaker type.

        Each recording is evaluated against the resolved and unresolved DP waveforms of the
        attended and the ignored voice with the cross-correlation, MSC and PSD methods of
        `DPOAEProcessor`. Results are written to
        `res_path/<speaker_type>_Speaker/<Successful|Unsuccessful>_attention/Target_<target>/<harm_type>/<Attended|Ignored>_DPs/`,
        where baseline chapters go to 'Target_video_BL_Polarnacht_<voice>'. A chapter counts
        as successful if at least two (competing speaker) or one (single speaker)
        comprehension questions were answered correctly. Summary tables are written to
        `res_path` (`stats_analysis_DPOAE_<speaker_type>.csv`,
        `analysis_volume_<speaker_type>.csv` and, for in-ear recordings,
        `analysis_CQ_<speaker_type>.csv`).

        Parameters
        ----------
        speaker_type : str
            'Single' or 'Competing'.
        outofear : bool, optional
            True for out-of-ear control recordings: no comprehension questions are evaluated
            and all chapters count as successful. Default is False.
        """
        res_path = self.res_path / f'{speaker_type}_Speaker'
        res_path.mkdir(parents=True, exist_ok=True)

        passing_limit = 1 if speaker_type == 'Single' else 2

        columns_cq = ['Chapter', 'Target', 'Given A1', 'Given A2', 'Given A3', 'Correct A1', 'Correct A2', 'Correct A3',
                      'Number Correct Answers', 'Listening Effort']
        columns_stats = ['Chapter', 'Target', 'Stimulus Shortcut', 'Number Correct Answers']
        for role in ('ATTENDED', 'IGNORED'):
            for harm in ('resolved', 'unresolved'):
                columns_stats += [f'{role} Delay [frames] ({harm})', f'{role} Mag [a.u.] ({harm})',
                                  f'{role} SNR [dB] ({harm})', f'{role} Significant Peak ({harm})']
        columns_stats += ['Equipment Delay [frames]', 'Stimulus Path']
        columns_vol = ['Chapter', 'Target', 'Mean Volume Stimulus [dB SPL]', 'Max Volume Stimulus [dB SPL]']

        target_info = self.load_target_info(speaker_type)

        if not outofear:
            exp_info = self.load_experiment_info(speaker_type)
            given_answers = self.get_given_answers(exp_info)
            correct_answers = self.get_correct_answers(target_info)
            count_correct_answers = np.sum(self.compare_answers(given_answers, correct_answers), axis=0)
            listening_effort = self.get_listening_effort(exp_info)

        oae_recs = self.get_recordings(speaker_type)

        targets = target_info['Speaker'] if speaker_type == 'Single' else target_info['Target']
        stim_paths = target_info['Stimulus']
        chapters = [int(c) for c in target_info['Chapter']]

        dpoae_bandwidths = self.load_dpoae_bandwidths()

        print(f'\nAnalyzing DPOAE recordings for {speaker_type} Speaker...')

        list_stats, list_vol = [], []
        for i, chapter in enumerate(chapters):
            print(f'Chapter {chapter}...')

            target = targets[i].strip()
            stim_path = stim_paths[i].strip().replace('\\', '/')
            oae_rec = oae_recs[i]

            mean_vol_stim, max_vol_stim = self.estimate_level(oae_rec)

            if outofear:
                sub_path = res_path / 'Successful_attention'
                no_correct_answers = None
            else:
                no_correct_answers = count_correct_answers[i]
                sub_path = res_path / ('Successful_attention' if no_correct_answers >= passing_limit else 'Unsuccessful_attention')

            stim_name, wf_dp_attended, wf_dp_ignored = self.load_dp_waveforms(target, chapter, speaker_type, stim_path)

            if target == 'female':
                atd_speaker, ign_speaker = 'female', 'male'
            elif target == 'male':
                atd_speaker, ign_speaker = 'male', 'female'
            elif target == 'video':
                atd_speaker, ign_speaker = ('female', 'male') if 'Polarnacht_female' in stim_name else ('male', 'female')
            else:
                raise ValueError('Invalid target!')

            subfolder = f'Target_{target}_BL_Polarnacht_{atd_speaker}' if target == 'video' else f'Target_{target}'
            save_path = sub_path / subfolder
            save_path.mkdir(parents=True, exist_ok=True)

            stimuli, _ = librosa.load(self.exp_path / stim_path, sr=self.fs, mono=False)

            # remove the equipment delay from the recording, then check that it is gone
            eq_delay = self.compute_equipment_delay([stimuli[0], stimuli[1]], oae_rec, save_path / f'Ch{chapter}_eq_delay')
            oae_rec = oae_rec[int(eq_delay):]
            eq_delay_check = np.round(self.compute_equipment_delay([stimuli[0], stimuli[1]], oae_rec))
            if eq_delay_check > 50:
                raise ValueError(f'Equipment delay not removed correctly for chapter {chapter}! '
                                 f'Expected ~0, got {eq_delay_check} frames.')
            eq_delay = 0

            n = min(stimuli.shape[1], wf_dp_attended.shape[1], wf_dp_ignored.shape[1], len(oae_rec))
            wf_dp_attended, wf_dp_ignored, oae_rec = wf_dp_attended[:, :n], wf_dp_ignored[:, :n], oae_rec[:n]

            attended_stats, ignored_stats = [], []
            for h_idx, harm_type in enumerate(['resolved', 'unresolved']):
                save_path_attended = save_path / harm_type / 'Attended_DPs'
                save_path_ignored = save_path / harm_type / 'Ignored_DPs'
                save_path_attended.mkdir(parents=True, exist_ok=True)
                save_path_ignored.mkdir(parents=True, exist_ok=True)

                dp_attended, dp_ignored = wf_dp_attended[h_idx], wf_dp_ignored[h_idx]

                # cross-correlation
                attended_stats += self.compute_DPOAE_crosscorr(dp_attended, oae_rec, eq_delay, save_path_attended / f'Ch{chapter}_')
                ignored_stats += self.compute_DPOAE_crosscorr(dp_ignored, oae_rec, eq_delay, save_path_ignored / f'Ch{chapter}_')

                band_attended = dpoae_bandwidths[atd_speaker][harm_type]
                band_ignored = dpoae_bandwidths[ign_speaker][harm_type]

                # MSC
                dict_msc_atd = self.compute_DPOAE_MSC(dp_attended, oae_rec, band_attended)
                dict_msc_ign = self.compute_DPOAE_MSC(dp_ignored, oae_rec, band_ignored)
                self.plot_DPOAE_MSC(dict_msc_atd, title=f'{harm_type.capitalize()} Target {target} Attended DPs',
                                    save_path=save_path_attended / f'Ch{chapter}_MSC_plot.png')
                self.plot_DPOAE_MSC(dict_msc_ign, title=f'{harm_type.capitalize()} Target {target} Ignored DPs',
                                    save_path=save_path_ignored / f'Ch{chapter}_MSC_plot.png')
                np.save(save_path_attended / f'Ch{chapter}_MSC_results.npy', dict_msc_atd, allow_pickle=True)
                np.save(save_path_ignored / f'Ch{chapter}_MSC_results.npy', dict_msc_ign, allow_pickle=True)

                # PSD
                dict_psd_atd = self.compute_DPOAE_PSD(f'{harm_type}_Target{target[0].capitalize()}_attended', oae_rec,
                                                      band_attended, save_path_attended / f'Ch{chapter}_')
                dict_psd_ign = self.compute_DPOAE_PSD(f'{harm_type}_Target{target[0].capitalize()}_ignored', oae_rec,
                                                      band_ignored, save_path_ignored / f'Ch{chapter}_')
                np.save(save_path_attended / f'Ch{chapter}_PSD_results.npy', dict_psd_atd, allow_pickle=True)
                np.save(save_path_ignored / f'Ch{chapter}_PSD_results.npy', dict_psd_ign, allow_pickle=True)

            list_stats.append([chapter, target, stim_name, no_correct_answers] + attended_stats + ignored_stats
                              + [eq_delay, stim_path])
            list_vol.append([chapter, target, mean_vol_stim, max_vol_stim])

        if not outofear:
            c = np.array(given_answers).shape[1]
            list_cq = np.concatenate((np.array(list_stats)[:, :2], np.array(given_answers).T, np.array(correct_answers).T,
                                      np.array(count_correct_answers).reshape(c, 1),
                                      np.array(listening_effort).reshape(c, 1)), axis=1)
            self.write_results_csv(columns_cq, list_cq, self.res_path / f'analysis_CQ_{speaker_type}.csv')

        self.write_results_csv(columns_stats, list_stats, self.res_path / f'stats_analysis_DPOAE_{speaker_type}.csv')
        self.write_results_csv(columns_vol, list_vol, self.res_path / f'analysis_volume_{speaker_type}.csv')
