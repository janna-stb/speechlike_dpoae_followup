"""Generation of speech-derived harmonic stimuli and multiband competing-speaker stimuli (JS26).

`stimulus_generator` derives, for every chapter of a (pitch-shifted) audiobook, waveforms that
follow the n-th harmonic of the speaker's fundamental frequency, as well as the expected
distortion-product (DP) waveforms. `multiband_generator` combines the harmonic waveforms of
the story and the competing speaker into the stereo stimuli presented during the experiment,
and mixes the corresponding audiobook chapters.
"""

import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
import scipy.signal as sig
import soundfile as sf

from plot_stimulus_correlations import stimuli_correlation, plot_GA
from running_mean import RunningStatsArray

# shared signal-processing helpers live in ../analysis
sys.path.append(str(Path(__file__).resolve().parents[1] / 'analysis'))
from functions_filter import butter_filter  # noqa: E402


class stimulus_generator:
    """Generate harmonic and DP waveforms from the audiobook chapters in one folder.

    For each chapter, the fundamental waveform is obtained by band-pass filtering the audio
    around its median f0. Its Hilbert phase, multiplied by n, gives the waveform of the n-th
    harmonic. Waveforms are created for the harmonic numbers n1 and n2 and for the
    corresponding DPs 2*n1 - n2 and 2*n2 - n1.

    Parameters
    ----------
    n1 : int or list of int
        Harmonic number(s) of the lower stimulus component(s).
    n2 : int or list of int
        Harmonic number(s) of the upper stimulus component(s); same length as `n1`.
    source_dir : str or Path
        Folder containing the audiobook chapters (.wav), e.g. `.../STORY_Male/f0_to_140Hz`.
    target_dir : str or Path
        Folder the harmonic waveforms are saved to, e.g. `.../Stimuli/STORY_Male`. The DP
        waveforms are saved to `<target_dir parent>/DP_<target_dir name>`.
    fs : int, optional
        Sampling frequency in Hz. Default is 44100.
    """

    def __init__(self, n1, n2, source_dir, target_dir, fs=44100):
        self.fs = fs
        self.n1 = np.atleast_1d(np.array(n1))
        self.n2 = np.atleast_1d(np.array(n2))
        if len(self.n1) != len(self.n2):
            raise ValueError('n1 and n2 must have the same length.')

        self.dp = self.distortion_product(self.n1, self.n2)
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def distortion_product(self, n1, n2):
        """Harmonic numbers of the lower (2*n1 - n2) and upper (2*n2 - n1) DPs.

        Parameters
        ----------
        n1, n2 : int or array-like of int
            Harmonic numbers of the two stimulus components.

        Returns
        -------
        np.ndarray or list
            [lower DPs..., upper DPs...].
        """
        if isinstance(n1, Iterable) and isinstance(n2, Iterable):
            return np.concatenate((2 * np.array(n1) - np.array(n2), 2 * np.array(n2) - np.array(n1)))
        return [2 * n1 - n2, 2 * n2 - n1]

    def generate_sine_wave(self, frequency, duration, nrm=0.01):
        """Return a sine wave of `frequency` Hz, `duration` s and amplitude `nrm`."""
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        return nrm * np.sin(2 * np.pi * frequency * t)

    def save_wave(self, wave, filename):
        """Save a waveform as .wav file, creating the folder if necessary."""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        sf.write(filename, wave, self.fs)

    def zscore_norm(self, data):
        """Return `data` normalised to zero mean and unit standard deviation."""
        return (data - np.mean(data)) / np.std(data)

    def load_source_audio(self):
        """List all .wav files in `source_dir` (stored in `self.source_files`)."""
        self.source_files = sorted(self.source_dir.glob('*.wav'))
        if not self.source_files:
            raise FileNotFoundError('No .wav files found in the source directory.')

    def determine_f0(self, signal, return_median=True):
        """Estimate the fundamental frequency of a speech signal with Praat (parselmouth).

        Parameters
        ----------
        signal : np.ndarray
            The audio signal.
        return_median : bool, optional
            If True, return median and standard deviation of f0 in voiced frames; otherwise
            the f0 values of all voiced frames. Default is True.

        Returns
        -------
        tuple or np.ndarray
            (median, std) in Hz, or the voiced f0 values (NaN if no frame is voiced).
        """
        pitch = parselmouth.Sound(signal, sampling_frequency=self.fs).to_pitch(time_step=0.01)
        f0 = pitch.selected_array['frequency']
        f0_voiced = f0[f0 > 0]
        if len(f0_voiced) == 0:
            return np.nan
        if return_median:
            return np.median(f0_voiced), np.std(f0_voiced)
        return f0_voiced

    def estimate_stim_frequency(self, signal):
        """Estimate the frequency of a generated waveform from its instantaneous frequency
        (Hilbert transform after band-pass filtering 100-5000 Hz).

        Parameters
        ----------
        signal : np.ndarray
            The waveform.

        Returns
        -------
        tuple
            Median and standard deviation of the instantaneous frequency in Hz.
        """
        sos = sig.butter(4, [100, 5000], btype='bandpass', fs=self.fs, output='sos')
        phase = np.unwrap(np.angle(sig.hilbert(sig.sosfilt(sos, signal))))
        inst_freq = np.diff(phase) * self.fs / (2 * np.pi)
        inst_freq = inst_freq[(inst_freq > 0) & (inst_freq < self.fs / 2)]
        return np.median(inst_freq), np.std(inst_freq)

    def gen_fundamental_waveform(self, x, f0_mean, bandwidth):
        """Band-pass filter an audio signal around its fundamental frequency.

        Parameters
        ----------
        x : np.ndarray
            The audio signal.
        f0_mean : float
            Centre frequency in Hz (median f0 of the signal).
        bandwidth : float
            Filter bandwidth in Hz; the passband is f0_mean ± bandwidth/2.

        Returns
        -------
        np.ndarray
            The z-scored fundamental waveform.
        """
        lowcut = np.round(f0_mean - bandwidth / 2, 1)
        highcut = np.round(f0_mean + bandwidth / 2, 1)
        return self.zscore_norm(butter_filter(x, [lowcut, highcut], 'bandpass', fs=self.fs))

    def gen_stimuli_hilbert(self, n, input_wf, nrm=0.1, lowpass_filter=True):
        """Generate the waveform of the n-th harmonic from the fundamental waveform.

        The fundamental waveform is extended by its time-reversed copy to reduce edge effects
        of the Hilbert transform; the harmonic is amp * cos(n * phase) of the analytic signal.

        Parameters
        ----------
        n : int
            Harmonic number.
        input_wf : np.ndarray
            Fundamental waveform (see `gen_fundamental_waveform`).
        nrm : float, optional
            Amplitude scaling. Default is 0.1.
        lowpass_filter : bool, optional
            Apply a 7.5 kHz low-pass filter, so that recordings can later be downsampled to
            15 kHz without aliasing. Default is True.

        Returns
        -------
        np.ndarray
            The harmonic waveform without DC offset.
        """
        fund_wf = np.nan_to_num(input_wf)
        fund_wf = np.concatenate((fund_wf[::-1], fund_wf))

        wf_hilbert = sig.hilbert(fund_wf)
        amp = np.abs(wf_hilbert[len(input_wf):])
        phase = np.unwrap(np.angle(wf_hilbert[len(input_wf):]))

        stim = nrm * np.nan_to_num(amp * np.cos(phase * n))
        if lowpass_filter:
            stim = butter_filter(stim, 7500, 'lowpass', fs=self.fs)
        return stim - np.mean(stim)

    def create_pure_tone_stimuli(self, f1, f2, duration=120):
        """Create the stereo pure-tone stimulus (f1, f2) and the waveform of its lower DP.

        Saved as `stereo_pt.wav` and `dp_pt.wav` in the parent folder of `target_dir`.

        Parameters
        ----------
        f1, f2 : float
            Frequencies of the two tones in Hz.
        duration : float, optional
            Duration in s. Default is 120 s.
        """
        pt_dp = self.distortion_product(f1, f2)[0]  # lower DP, 2*f1 - f2
        stereo_stim = np.vstack((self.generate_sine_wave(f1, duration), self.generate_sine_wave(f2, duration))).T
        self.save_wave(stereo_stim, self.target_dir.parent / 'stereo_pt.wav')
        self.save_wave(self.generate_sine_wave(pt_dp, duration), self.target_dir.parent / 'dp_pt.wav')

    def create_stimuli(self, lowpass_filter=True, bw='f0_std'):
        """Create the harmonic and DP waveforms of all chapters in `source_dir`.

        Waveforms are saved as `<chapter>_n<k>.wav` in `target_dir/<source folder name>` (DP
        waveforms in `DP_<target_dir name>/<source folder name>`); existing files are skipped.
        Frequencies of the generated waveforms are appended to `stimuli_info.csv`, and the
        cross-correlations between stimulus and DP waveforms are computed by
        `determine_stimulus_correlations`.

        Parameters
        ----------
        lowpass_filter : bool, optional
            Passed to `gen_stimuli_hilbert`. Default is True.
        bw : 'f0_std' or float, optional
            Bandwidth of the fundamental waveform: the standard deviation of f0 of each
            chapter, or a fixed value in Hz. Default is 'f0_std'.
        """
        self.load_source_audio()

        csv_stimuli = []
        stim_columns = ['filename', 'f0_mean', 'f0_std', 'harm', 'harm_freq [Hz]', 'harm_std [Hz]']

        for audio_file in self.source_files:
            audio_name = audio_file.stem
            out_path = self.target_dir / audio_file.parent.stem
            out_path_dp = self.target_dir.parent / f'DP_{self.target_dir.stem}' / audio_file.parent.stem

            audio_array, _ = librosa.load(audio_file, sr=self.fs)
            audio_array = audio_array / np.max(np.abs(audio_array))

            f0_median, f0_std = self.determine_f0(audio_array)
            if bw == 'f0_std':
                bw_fund_wf = f0_std
            elif isinstance(bw, (int, float)):
                bw_fund_wf = bw
            else:
                raise ValueError("Invalid bandwidth option. Choose 'f0_std' or a number in Hz.")

            fund_wf = self.gen_fundamental_waveform(audio_array, f0_median, bw_fund_wf)
            f0_check, _ = self.estimate_stim_frequency(fund_wf)
            if np.abs(f0_check - f0_median) > 10:
                print(f'Warning: Estimated f0 of fundamental waveform ({f0_check:.2f} Hz) deviates from original f0 '
                      f'({f0_median:.2f} Hz) for {audio_name}. Check the generated waveform.')

            for harms, save_to in zip([self.n1, self.n2, self.dp], [out_path, out_path, out_path_dp]):
                for n in harms:
                    filename = save_to / f'{audio_name}_n{n}.wav'
                    if filename.exists():
                        print(f'\n{Path(*filename.parts[-3:])} already exists, skipping...')
                        continue
                    wf = self.gen_stimuli_hilbert(n, fund_wf, lowpass_filter=lowpass_filter)
                    freq_stim, std_stim = self.estimate_stim_frequency(wf)
                    self.save_wave(wf, filename)
                    csv_stimuli.append([audio_name, f0_median, f0_std, n, freq_stim, std_stim])

            self.determine_stimulus_correlations(out_path_dp, out_path, audio_name)
            csv_stimuli.append(['-' * 10] * 6)  # separator between chapters

        csv_path = out_path / 'stimuli_info.csv'
        pd.DataFrame(csv_stimuli, columns=stim_columns).to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)

    def determine_stimulus_correlations(self, dp_path, audio_path, audio_name):
        """Cross-correlate the lower DP waveforms with the stimulus waveforms of one chapter.

        A significant correlation would indicate that the DP waveform is present in the
        stimulus itself. Plots are saved to `audio_path/Plots/`, results appended to
        `audio_path/stimuli_correlations.csv`.

        Parameters
        ----------
        dp_path : Path
            Folder of the DP waveforms.
        audio_path : Path
            Folder of the harmonic waveforms.
        audio_name : str
            Chapter name (file stem).
        """
        plot_path = audio_path / 'Plots'
        plot_path.mkdir(parents=True, exist_ok=True)

        csv_correlations = []
        for n1, n2, dp in zip(self.n1, self.n2, self.dp):
            stim_n1 = librosa.load(audio_path / f'{audio_name}_n{n1}.wav', sr=self.fs)[0]
            stim_n2 = librosa.load(audio_path / f'{audio_name}_n{n2}.wav', sr=self.fs)[0]
            stim_dp = librosa.load(dp_path / f'{audio_name}_n{dp}.wav', sr=self.fs)[0]

            sig_peak, _ = stimuli_correlation(dp, stim_dp, stim_n1, stim_n2,
                                              filename=plot_path / f'corr_{audio_name}_n{n1}n{n2}_dp{dp}.png', fs=self.fs)
            csv_correlations.append([audio_name, n1, n2, dp, sig_peak])

        csv_path = audio_path / 'stimuli_correlations.csv'
        pd.DataFrame(csv_correlations, columns=['filename', 'n1', 'n2', 'dp', 'sig_peak']).to_csv(
            csv_path, mode='a', header=not csv_path.exists(), index=False)


class multiband_generator:
    """Combine the harmonic waveforms of two books (story and competing speaker) into the
    stereo stimuli of the competing-speaker experiment, and mix the audiobook chapters.

    Parameters
    ----------
    stim_directory : str or Path
        Folder containing the harmonic waveforms (and the DP waveforms in 'DP_' folders).
    bookA : str
        Sub-folder of `stim_directory` with the harmonic waveforms of book A (story), e.g.
        'STORY_Male/f0_to_140Hz'.
    bookB : str, optional
        Sub-folder with the harmonic waveforms of book B (competing speaker).
    fs : int, optional
        Sampling frequency in Hz. Default is 44100.
    """

    def __init__(self, stim_directory, bookA, bookB=None, fs=44100):
        self.stim_directory = Path(stim_directory)
        self.bookA = bookA
        self.bookB = bookB
        self.fs = fs

    def distortion_product(self, n1, n2):
        """Harmonic number(s) of the lower DP, 2*n1 - n2."""
        if isinstance(n1, Iterable) and isinstance(n2, Iterable):
            return 2 * np.array(n1) - np.array(n2)
        return 2 * n1 - n2

    def save_wave(self, wave, filename):
        """Save a waveform as .wav file, creating the folder if necessary."""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        sf.write(filename, wave, self.fs)

    def extract_chapter_number(self, filename):
        """Return the two-digit chapter number in a file name (without the '_n<k>' suffix), or None."""
        base = re.sub(r'_n\d+$', '', filename)
        match = re.findall(r'(?<!\d)(\d{2})(?!\d)', base)
        return match[0] if match else None

    def _target_suffix(self):
        """Folder name describing both books and their f0, e.g. 'STORY_140Hz_NOISE_180Hz'."""
        a_folder, a_shift = self.bookA.split('/')
        b_folder, b_shift = self.bookB.split('/')
        return f"{a_folder.split('_')[0]}_{a_shift.split('_')[2]}_{b_folder.split('_')[0]}_{b_shift.split('_')[2]}"

    def load_stim_files(self, book):
        """List the waveform files of a book: all .wav files in `stim_directory/book`, or, if
        `book` contains no '/', all files in `stim_directory` starting with `book`."""
        if len(book.split('/')) == 1:
            stim_files = sorted(self.stim_directory.glob(f'{book}*.wav'))
        else:
            stim_files = sorted((self.stim_directory / book).glob('*.wav'))
        if not stim_files:
            raise FileNotFoundError('No .wav files found in the stimulus directory.')
        return stim_files

    def create_multiband_competingspeaker(self, bookA, bookB, A_n1, A_n2, B_n1, B_n2, nrm=0.01, target_dir=None):
        """Create the stereo competing-speaker stimuli of all chapters.

        For each chapter, the n1 waveforms of both books are summed for the left channel and
        the n2 waveforms for the right channel; each channel is normalised to a peak of `nrm`.
        If the DP waveforms are available, the cross-correlations of the stimuli with the DP
        waveforms of both books are computed (plots and `multiband_correlations_*.csv`) and
        their grand averages plotted.

        Parameters
        ----------
        bookA, bookB : str
            Sub-folders of `stim_directory` with the harmonic waveforms of the two books.
        A_n1, A_n2, B_n1, B_n2 : int or list of int
            Harmonic numbers of the left (n1) and right (n2) channel for book A and book B.
        nrm : float, optional
            Peak amplitude of each channel. Default is 0.01.
        target_dir : str or Path, optional
            Output folder. Default is
            `<stim_directory parent>/Uncalibrated_Stimuli/competing_stimuli_<A>_<B>/<suffix>`.
        """
        if target_dir is None:
            target_dir = (self.stim_directory.parent / 'Uncalibrated_Stimuli'
                          / f"competing_stimuli_{self.bookA.split('/')[0]}_{self.bookB.split('/')[0]}" / self._target_suffix())
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        A_n1, A_n2, B_n1, B_n2 = (list(np.atleast_1d(x)) for x in (A_n1, A_n2, B_n1, B_n2))

        stim_bookA = self.load_stim_files(bookA)
        stim_bookB = self.load_stim_files(bookB)

        skip_dp_corr = len(bookA.split('/')) == 1 or len(bookB.split('/')) == 1
        if not skip_dp_corr:
            dp_wfs_bookA = self.load_stim_files('DP_' + bookA)
            dp_wfs_bookB = self.load_stim_files('DP_' + bookB)

        chapters_A = sorted(set(self.extract_chapter_number(file.stem) for file in stim_bookA))
        chapters_B = sorted(set(self.extract_chapter_number(file.stem) for file in stim_bookB))
        if chapters_A != chapters_B:
            raise ValueError('Chapters in book A and book B do not match.')

        GA_env_A = defaultdict(RunningStatsArray)
        GA_env_B = defaultdict(RunningStatsArray)

        harms_A = ''.join(f'n{n}' for n in A_n1) + '_' + ''.join(f'n{n}' for n in A_n2)
        harms_B = ''.join(f'n{n}' for n in B_n1) + '_' + ''.join(f'n{n}' for n in B_n2)

        for chapter in chapters_A:
            chapter_files_A = [f for f in stim_bookA if self.extract_chapter_number(f.stem) == chapter]
            chapter_files_B = [f for f in stim_bookB if self.extract_chapter_number(f.stem) == chapter]
            if chapter is None:
                chapter = 'Intro'
            base_name = 'Ch_' + chapter

            n1_files_A = [f for f in chapter_files_A if any(f'n{n}' in f.stem for n in A_n1)]
            n2_files_A = [f for f in chapter_files_A if any(f'n{n}' in f.stem for n in A_n2)]
            n1_files_B = [f for f in chapter_files_B if any(f'n{n}' in f.stem for n in B_n1)]
            n2_files_B = [f for f in chapter_files_B if any(f'n{n}' in f.stem for n in B_n2)]
            if len(n1_files_A) != len(n2_files_A) or len(n1_files_B) != len(n2_files_B):
                raise ValueError(f'Number of n1 and n2 files do not match for chapter {chapter}.')

            multiband_n1 = np.array([])
            multiband_n2 = np.array([])
            for n1_file_A, n1_file_B, n2_file_A, n2_file_B in zip(n1_files_A, n1_files_B, n2_files_A, n2_files_B):
                stim_n1_A, _ = librosa.load(n1_file_A, sr=self.fs)
                stim_n2_A, _ = librosa.load(n2_file_A, sr=self.fs)
                stim_n1_B, _ = librosa.load(n1_file_B, sr=self.fs)
                stim_n2_B, _ = librosa.load(n2_file_B, sr=self.fs)

                # zero-pad the shorter book
                n1_len, n2_len = max(len(stim_n1_A), len(stim_n1_B)), max(len(stim_n2_A), len(stim_n2_B))
                stim_n1_A, stim_n1_B = (np.pad(s, (0, n1_len - len(s))) for s in (stim_n1_A, stim_n1_B))
                stim_n2_A, stim_n2_B = (np.pad(s, (0, n2_len - len(s))) for s in (stim_n2_A, stim_n2_B))

                if multiband_n1.shape[0] == 0 and multiband_n2.shape[0] == 0:
                    multiband_n1 = stim_n1_A + stim_n1_B
                    multiband_n2 = stim_n2_A + stim_n2_B
                else:
                    multiband_n1 += stim_n1_A + stim_n1_B
                    multiband_n2 += stim_n2_A + stim_n2_B

            multiband_n1 = nrm * multiband_n1 / np.max(np.abs(multiband_n1))
            multiband_n2 = nrm * multiband_n2 / np.max(np.abs(multiband_n2))

            if chapter == 'Intro':
                filename = target_dir / (f'Intro_{harms_A}.wav' if harms_A == harms_B else f'Intro_BookA_{harms_A}_BookB_{harms_B}.wav')
            else:
                filename = target_dir / f"{base_name}_{bookA.split('/')[0]}_{harms_A}_{bookB.split('/')[0]}_{harms_B}.wav"
            self.save_wave(np.vstack((multiband_n1, multiband_n2)).T, filename)

            if not skip_dp_corr:
                dp_A = self.distortion_product(A_n1, A_n2)
                dp_B = self.distortion_product(B_n1, B_n2)
                dp_files_A = [f for f in dp_wfs_bookA if any(f'n{d}' in f.stem for d in dp_A) and f'_{chapter}' in f.stem]
                dp_files_B = [f for f in dp_wfs_bookB if any(f'n{d}' in f.stem for d in dp_B) and f'_{chapter}' in f.stem]

                sig_peak = []
                for i, (dp_file_A, dp_file_B) in enumerate(zip(dp_files_A, dp_files_B)):
                    wf_dp_A, _ = librosa.load(dp_file_A, sr=self.fs)
                    wf_dp_B, _ = librosa.load(dp_file_B, sr=self.fs)
                    sig_A, res_A = stimuli_correlation(dp_A[i], wf_dp_A, multiband_n1, multiband_n2,
                                                       filename=target_dir / f"corr_{base_name}_{bookA.split('/')[0]}_dp{dp_A[i]}.png", fs=self.fs)
                    sig_B, res_B = stimuli_correlation(dp_B[i], wf_dp_B, multiband_n1, multiband_n2,
                                                       filename=target_dir / f"corr_{base_name}_{bookB.split('/')[0]}_dp{dp_B[i]}.png", fs=self.fs)
                    sig_peak.append((sig_A, sig_B))
                    GA_env_A[f'{dp_A[i]}'].push(res_A)
                    GA_env_B[f'{dp_B[i]}'].push(res_B)

                k = len(sig_peak)
                multiband_columns = ['filename', 'book A', 'book B', 'n1 (book A)', 'n2 (book A)', 'dp (book A)',
                                     'n1 (book B)', 'n2 (book B)', 'dp (book B)', 'sig_peak_A', 'sig_peak_B']
                multiband_data = np.array([[base_name] * k, [bookA] * k, [bookB] * k,
                                           ['n'.join(map(str, A_n1))] * k, ['n'.join(map(str, A_n2))] * k, list(dp_A),
                                           ['n'.join(map(str, B_n1))] * k, ['n'.join(map(str, B_n2))] * k, list(dp_B),
                                           [res[0] for res in sig_peak], [res[1] for res in sig_peak]]).T
                csv_path = target_dir.parent / f'multiband_correlations_{self._target_suffix()}.csv'
                pd.DataFrame(multiband_data, columns=multiband_columns).to_csv(csv_path, mode='a', header=not csv_path.exists(),
                                                                              index=False)

        for key_A, key_B in zip(GA_env_A.keys(), GA_env_B.keys()):
            plot_GA(key_A, GA_env_A[key_A].mean, target_dir / f"GA_{bookA.split('/')[0]}_dp{key_A}.png")
            plot_GA(key_B, GA_env_B[key_B].mean, target_dir / f"GA_{bookB.split('/')[0]}_dp{key_B}.png")

    def mix_books(self, audio_path, bookA, bookB, target_dir=None, nrm=0.2, offset=0):
        """Mix the audiobook chapters of book A and book B (presented together with the stimuli).

        Both books are low-pass filtered at 7.5 kHz (for the evaluation of ipsilateral
        recordings), normalised, zero-padded to equal length and summed; book B starts with
        `offset/2` s of silence followed by a linear fade-in until `offset` s.

        Parameters
        ----------
        audio_path : str or Path
            Folder containing the (pitch-shifted) audiobooks.
        bookA, bookB : str
            Sub-folders of `audio_path` with the chapters of the two books (or file-name
            prefixes if they contain no '/').
        target_dir : str or Path, optional
            Output folder. Default is
            `<stim_directory parent>/Mixed_Audiobooks_<A>_<B>/<suffix>`.
        nrm : float or sequence of 3 floats, optional
            Peak amplitude of both books and of the mix, or [book A, book B, mix]. Default is 0.2.
        offset : float, optional
            Delay of book B in s (see above). Default is 0.
        """
        audio_path = Path(audio_path)
        if target_dir is None:
            target_dir = (self.stim_directory.parent / f"Mixed_Audiobooks_{self.bookA.split('/')[0]}_{self.bookB.split('/')[0]}"
                          / self._target_suffix())
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if len(bookA.split('/')) == 1 or len(bookB.split('/')) == 1:
            source_files_A = sorted(audio_path.glob(f'{bookA}*.wav'))
            source_files_B = sorted(audio_path.glob(f'{bookB}*.wav'))
        else:
            source_files_A = sorted((audio_path / bookA).glob('*.wav'))
            source_files_B = sorted((audio_path / bookB).glob('*.wav'))
        if len(source_files_A) != len(source_files_B):
            raise ValueError(f'Number of audio files in {bookA} and {bookB} do not match.')

        chapters_A = sorted(set(self.extract_chapter_number(file.stem) for file in source_files_A))
        chapters_B = sorted(set(self.extract_chapter_number(file.stem) for file in source_files_B))
        if chapters_A != chapters_B:
            raise ValueError('Chapters in book A and book B do not match.')

        if isinstance(nrm, (list, tuple)):
            nrm_A, nrm_B, nrm_mixed = nrm
        else:
            nrm_A = nrm_B = nrm_mixed = nrm

        for chapter in chapters_A:
            file_A = next(f for f in source_files_A if self.extract_chapter_number(f.stem) == chapter)
            file_B = next(f for f in source_files_B if self.extract_chapter_number(f.stem) == chapter)

            audio_A = butter_filter(librosa.load(file_A, sr=self.fs)[0], 7500, 'lowpass', fs=self.fs)
            audio_B = butter_filter(librosa.load(file_B, sr=self.fs)[0], 7500, 'lowpass', fs=self.fs)
            audio_A = nrm_A * audio_A / np.max(np.abs(audio_A))
            audio_B = nrm_B * audio_B / np.max(np.abs(audio_B))

            n = max(len(audio_A), len(audio_B))
            audio_A, audio_B = np.pad(audio_A, (0, n - len(audio_A))), np.pad(audio_B, (0, n - len(audio_B)))

            if offset > 0:
                midpoint = int(offset / 2 * self.fs)
                audio_B[:midpoint] = 0
                audio_B[midpoint:int(offset * self.fs)] *= np.linspace(0, 1, midpoint)

            mixed_audio = audio_A + audio_B
            mixed_audio = nrm_mixed * mixed_audio / np.max(np.abs(mixed_audio))

            if chapter is None:
                filename = target_dir / 'Intro_mixed.wav'
            else:
                filename = target_dir / f"Ch_{chapter}_{bookA.split('/')[0]}_{bookB.split('/')[0]}.wav"
            self.save_wave(mixed_audio, filename)
