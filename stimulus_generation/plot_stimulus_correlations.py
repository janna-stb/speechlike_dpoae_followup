"""Cross-correlation checks between generated stimuli and DP waveforms.

Unlike the functions in `analysis/plot_grand_averages.py`, the functions in this module
expect lags in ms.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# shared signal-processing helpers live in ../analysis
sys.path.append(str(Path(__file__).resolve().parents[1] / 'analysis'))
from functions_crosscorr import cross_corr  # noqa: E402

def find_peak_amplitude(env, lag, border=15):
    """Finds the peak amplitude in the envelope and its corresponding lag.

    Parameters
    ----------
    env : array
        The envelope of the cross-correlation.
    lag : array
        The lags corresponding to the envelope in ms.
    border : float, optional
        Restrict the search to lags within [-border, border] ms. Default is 15.

    Returns
    -------
    lag_peak : float
        The lag at which the peak occurs.
    mag_peak : float
        The magnitude of the peak.
    """
    idx_min = np.argmin(np.abs(lag + border))
    idx_max = np.argmin(np.abs(lag - border))
    env_range = env[idx_min:idx_max + 1]
    mag_peak, idx = np.max(env_range), np.argmax(env_range)
    idx = idx + idx_min
    lag_peak = lag[idx]
    
    return lag_peak, mag_peak

def determine_noise_level(env, lag, lower=70, upper=750):
    """Determines the noise level in the envelope by calculating the 95th percentile of the noise region.

    Parameters
    ----------
    env : array
        The envelope of the cross-correlation.
    lag : array
        The lags corresponding to the envelope in ms.
    lower, upper : float, optional
        The noise region comprises the lags with lower < |lag| < upper ms. Defaults are 70
        and 750 ms.

    Returns
    -------
    noise_level : float
        The 95th percentile of the noise region.
    env_noise : array
        The envelope values in the noise region.
    """
    idx_noise = np.where(((lag < -lower) & (lag > -upper)) | ((lag > lower) & (lag < upper)))[0]
    env_noise = env[idx_noise]
    noise_level = np.percentile(env_noise, 95)
    
    return noise_level, env_noise

def stimuli_correlation(ndp, dp, stim1, stim2, filename=None, fs=44100):
    """Calculates the cross-correlation between the stimuli and the distortion product waveform.

    Parameters
    ----------
    ndp : int
        The harmonic number of the distortion product
    dp : array
        The distortion product waveform
    stim1 : array
        The first stimulus waveform
    stim2 : array
        The second stimulus waveform
    filename : str
        The filename (including the path) where the plots should be saved
    fs : int
        The sampling frequency

    Returns
    -------
    sig_res : bool
        True if the cross-correlation with f1 + f2 has a peak within ±15 ms that exceeds the
        95th percentile of the noise region
    corr_results : list
        [lag, real, imag, envelope] of the cross-correlation with f1 + f2
    """
    # if directory for plots does not exist, create it
    if filename is not None:
        plot_dir = os.path.dirname(filename)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    stim_list = [stim1, stim2, stim1 + stim2]
    plt.figure(figsize=(10, 15))

    for i, stim in enumerate(stim_list):    
        lag, corr_real, corr_imag, env = cross_corr(dp, stim)
        lag_ms = lag / fs * 1000

        # find peak amplitude between -15 and 15 ms
        lag_peak_ms, mag = find_peak_amplitude(env, lag_ms, border=15)

        # determine if peak is significant
        noise_level, env_noise = determine_noise_level(env, lag_ms)

        if mag > noise_level:
            sig_res = True
        else:
            sig_res = False
                        
        # plot the correlations
        plt.subplot(3, 1, i+1)
        plt.plot(lag_ms, corr_real, label='real')
        plt.plot(lag_ms, corr_imag, label='imag')
        plt.plot(lag_ms, env, label='envelope', c='k')
        plt.plot(lag_ms, -env, c='k')
        if sig_res:
            plt.axvline(lag_peak_ms, c='k', ls='--', label=f'Sig. peak at {lag_peak_ms:.2f} ms')
        plt.legend(loc = 'upper right', ncol=2)
        plt.xlabel('Lag [ms]')
        plt.ylabel('Correlation')
        if i < 2:
            plt.title(f'Correlation of DP {ndp} vs. f{i+1}')
        else:
            plt.title(f'Correlation of DP {ndp} vs. f1 + f2')
        plt.grid()
        plt.xlim(-1000, 1000)
        plt.ylim(-max(env), max(env))
        plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
        plt.close()

    corr_results = [lag, corr_real, corr_imag, env]
    return sig_res, corr_results


def plot_GA(ndp, means, plot_path, fs=44100):
    """Plot the grand-average envelope of the stimulus-DP cross-correlations and compute
    its peak statistics.

    Parameters
    ----------
    ndp : int or str
        Harmonic number of the DP (used in the title).
    means : array
        Grand average in the form [lag (samples), real, imag, envelope].
    plot_path : str or Path
        Path of the figure.
    fs : int, optional
        Sampling frequency in Hz. Default is 44100.

    Returns
    -------
    peak_stats : tuple
        (lag of the peak in ms, peak magnitude, noise level, SNR in dB).
    """
    lag = means[0, :] # in frames
    lag_ms = lag / fs * 1000 # convert to ms
    env = means[3, :] # envelope of the cross-correlation
    
    lag_peak_ms, mag_peak = find_peak_amplitude(env, lag_ms)
    noise_level, env_noise = determine_noise_level(env, lag_ms)
    if mag_peak > max(env_noise):
        sig_peak = True
    else:
        sig_peak = False

    snr = 20 * np.log10(mag_peak / noise_level)

    plt.figure(figsize=(8, 5))
    plt.plot(lag_ms, env, label='env', color='black')
    plt.title(f'DP {ndp} - env')
    plt.xlim(-100, 100)
    plt.ylim(0, mag_peak * 2)
    if sig_peak:
        plt.axvline(lag_peak_ms, c='silver', ls='--', label=f'{lag_peak_ms:.2f} ms \nPeak significant', alpha=0.5)
    else:
        plt.axvline(lag_peak_ms, c='silver', ls='--', label=f'{lag_peak_ms:.2f} ms \nPeak not significant', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=1200)
    plt.close()

    peak_stats = (lag_peak_ms, mag_peak, noise_level, snr)
    
    return peak_stats
