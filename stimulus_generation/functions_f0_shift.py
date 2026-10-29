"""Shift the fundamental frequency (f0) of speech recordings with Praat's PSOLA.

The f0 contour of a recording is transformed in the log domain so that its mean matches a
target f0 while its relative variation is preserved, and the recording is resynthesised by
pitch-synchronous overlap-add (PSOLA, via parselmouth). Because the resynthesis does not hit
the target exactly, the shift is repeated with a global correction factor (closed loop).
"""

import sys
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

# shared signal-processing helpers live in ../analysis
sys.path.append(str(Path(__file__).resolve().parents[1] / 'analysis'))
from functions_filter import butter_filter  # noqa: E402


def get_f0_contour(sound):
    """Return the time points (s) and f0 values (Hz, 0 for unvoiced frames) of a Praat Sound."""
    pitch = sound.to_pitch(time_step=0.01)
    return pitch.xs(), pitch.selected_array['frequency']


def transform_f0_distribution(f0, target_mean):
    """Shift an f0 contour so that its geometric mean becomes `target_mean`.

    The transform is applied in the log domain and preserves the standard deviation of
    log(f0), i.e. the relative pitch variation. Unvoiced frames (f0 = 0) are left unchanged.

    Parameters
    ----------
    f0 : np.ndarray
        f0 contour in Hz.
    target_mean : float
        Target f0 in Hz.

    Returns
    -------
    np.ndarray
        The transformed contour.
    """
    voiced = f0 > 0
    if not np.any(voiced):
        return f0

    log_f0 = np.log(f0[voiced])
    mu, sigma = np.mean(log_f0), np.std(log_f0)
    log_f0_new = (log_f0 - mu) / sigma * sigma + np.log(target_mean)

    f0_new = f0.copy()
    f0_new[voiced] = np.exp(log_f0_new)
    return f0_new


def apply_psola(sound, times, f0_new):
    """Resynthesise a Praat Sound with a new f0 contour by overlap-add (PSOLA).

    Parameters
    ----------
    sound : parselmouth.Sound
        The recording.
    times : np.ndarray
        Time points of the contour in s.
    f0_new : np.ndarray
        New f0 values in Hz; unvoiced points (0) are skipped.

    Returns
    -------
    parselmouth.Sound
        The resynthesised recording.
    """
    manipulation = call(sound, 'To Manipulation', 0.01, 50, 800)  # time step, pitch floor, pitch ceiling
    pitch_tier = call(manipulation, 'Extract pitch tier')
    call(pitch_tier, 'Remove points between', 0, sound.duration)
    for t, f in zip(times, f0_new):
        if f > 0:
            call(pitch_tier, 'Add point', t, f)
    call([manipulation, pitch_tier], 'Replace pitch tier')
    return call(manipulation, 'Get resynthesis (overlap-add)')


def estimate_median_f0(signal, sr=44100):
    """Median f0 (Hz) of the voiced frames of a recording (Praat pitch estimation).

    Parameters
    ----------
    signal : parselmouth.Sound or np.ndarray
        The recording.
    sr : int, optional
        Sampling frequency in Hz, used for numpy input. Default is 44100.

    Returns
    -------
    float
        Median f0, or NaN if no frame is voiced.
    """
    sound = parselmouth.Sound(signal, sampling_frequency=sr) if isinstance(signal, np.ndarray) else signal
    f0 = sound.to_pitch(time_step=0.01).selected_array['frequency']
    f0 = f0[f0 > 0]
    return np.median(f0) if len(f0) else np.nan


def iterative_psola(sound, target_f0, n_iter=3):
    """Shift the f0 of a recording to `target_f0` with repeated PSOLA resynthesis.

    In each iteration, the contour is transformed (`transform_f0_distribution`) and
    resynthesised; the measured median f0 then defines a global correction factor, with
    which the transformed contour is resynthesised once more.

    Parameters
    ----------
    sound : parselmouth.Sound
        The recording.
    target_f0 : float
        Target f0 in Hz.
    n_iter : int, optional
        Number of iterations. Default is 3.

    Returns
    -------
    parselmouth.Sound
        The shifted recording.
    """
    snd = sound
    for i in range(n_iter):
        times, f0 = get_f0_contour(snd)
        f0_new = transform_f0_distribution(f0, target_f0)
        snd = apply_psola(snd, times, f0_new)

        measured = estimate_median_f0(snd)
        if np.isnan(measured):
            break
        print(f'Iter {i + 1}: f0={measured:.2f} Hz')

        snd = apply_psola(snd, times, f0_new * target_f0 / measured)
    return snd


def shift_f0_psola(x, target_f0, sr=44100):
    """Shift the f0 of a recording to `target_f0` and low-pass filter the result at 15 kHz.

    Parameters
    ----------
    x : np.ndarray
        The recording.
    target_f0 : float
        Target f0 in Hz.
    sr : int, optional
        Sampling frequency in Hz. Default is 44100.

    Returns
    -------
    snd_out : np.ndarray
        The shifted recording, shape (1, n_samples).
    f0_median : float
        Median f0 of the shifted recording in Hz.
    """
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    snd_out = iterative_psola(snd, target_f0, n_iter=3)
    snd_out = butter_filter(snd_out, 15000, 'lowpass', fs=sr)
    return snd_out, estimate_median_f0(snd_out, sr=sr)
