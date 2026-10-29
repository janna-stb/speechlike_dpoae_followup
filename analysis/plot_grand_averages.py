"""Peak and noise-floor estimation of cross-correlation envelopes, and grand-average plots."""

import numpy as np
import matplotlib.pyplot as plt

FS = 44100

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

def determine_noise_level(signal, lower=100, upper=500, return_noise_energy=False):
    """Determine the noise level of a cross-correlation envelope from the 95th percentile
    of its envelope values outside the expected DPOAE peak (the "noise region").

    Note that the lag axis of `signal` is in samples (converted with FS = 44100 Hz).

    The noise region is defined as the lags with absolute value between `lower` and
    `upper` ms (applied symmetrically to positive and negative lags), i.e. excluding both
    the near-zero-lag peak and the far tails of the signal.

    Parameters
    ----------
    signal : array
        The cross-correlation signal in the form [lag, real, imag, env] or [lag, env].
    lower : int, optional
        The lower bound of the noise region in ms. Default is 100 ms. Applied to positive
        and negative lags.
    upper : int, optional
        The upper bound of the noise region in ms. Default is 500 ms. Applied to positive
        and negative lags.
    return_noise_energy : bool, optional
        If True, return a dict with the mean noise floor and noise energy (computed from
        the real/imaginary components) instead of the percentile-based noise level.
        Requires `signal` to have shape (4, n). Default is False.

    Returns
    -------
    dict or tuple
        If `return_noise_energy` is True: {"env_floor": mean envelope noise, "energy_floor":
        mean squared real+imag noise energy}. Otherwise: (noise_level, env_noise), the 95th
        percentile of the noise region and the noise region's envelope values themselves.
    """
    if signal.shape[0] == 4:
        lag = signal[0, :]
        real = signal[1, :]
        imag = signal[2, :]
        env = signal[3, :]
    elif signal.shape[0] == 2:
        lag = signal[0, :]
        env = signal[1, :]

    # convert lower, upper to frames
    lower = lower / 1000 * FS
    upper = upper / 1000 * FS

    idx_noise = np.where(((lag < -lower) & (lag > -upper)) | ((lag > lower) & (lag < upper)))[0]

    env_noise = env[idx_noise]
    noise_level = np.percentile(env_noise, 95)

    if return_noise_energy:
        noise_floor = np.real(np.mean(env_noise))
        noise_energy = np.real(np.mean(real[idx_noise]**2 + imag[idx_noise]**2))
        return {"env_floor": noise_floor, "energy_floor": noise_energy}
    else:
        return noise_level, env_noise

def plot_GA_crosscorr(condition, means, plot_path, fs=44100):
    """Plot a condition's grand average cross-correlation envelope (marking the peak
    and whether it exceeds the noise ceiling) and compute its peak statistics.

    Parameters
    ----------
    condition : str
        The condition displayed. Only used for labeling the plot.
    means : array
        The grand average of the cross-correlation results in the form [lag, real, imag, env].
    plot_path : str or Path
        The path (including filename) the plot is saved to.
    fs : float, optional
        Sampling frequency in Hz used to convert lag frames to ms. Default is 44100.

    Returns
    -------
    peak_stats : list
        [lag_peak_ms, mag_peak, noise_floor, snr, z_peak] for this condition.
        `noise_floor` is the mean of the noise region. The SNR is computed relative to
        the mean noise floor, as for the single-trial SNR metric. `sig_peak` (whether
        mag_peak exceeds the max noise-region value, shown in the plot legend) is not
        itself included. `z_peak` is the distance of the peak from the noise floor in
        standard deviations of the noise region; a GA peak is considered reliable if it
        exceeds the max noise-region value and z_peak >= 4.
    """

    lag = means[0, :] # in frames
    lag_ms = lag / fs * 1000 # convert to ms
    env = means[3, :] # envelope of the cross-correlation
    
    lag_peak_ms, mag_peak = find_peak_amplitude(env, lag_ms)
    _, env_noise = determine_noise_level(means)
    if mag_peak > max(env_noise):
        sig_peak = True
    else:
        sig_peak = False

    # SNR relative to the mean noise floor, consistent with the SNR metric
    noise_floor = np.mean(env_noise)
    snr = 20 * np.log10(mag_peak / noise_floor)
    # distance of the peak from the noise floor in standard deviations of the noise region
    z_peak = (mag_peak - noise_floor) / np.std(env_noise)

    plt.figure(figsize=(8, 5))
    plt.plot(lag_ms, env, label='env', color='black')
    plt.title(f'DP {condition} - env')
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

    peak_stats = [lag_peak_ms, mag_peak, noise_floor, snr, z_peak]
    
    return peak_stats

def plot_GA_msc(title, freqs, means, plot_path, band_mask=None):
    """Plot a condition's grand average MSC (magnitude-squared coherence) spectrum, with
    the DPOAE frequency band shaded, and save both a full-range and band-zoomed version.

    Parameters
    ----------
    title : str
        Title displayed above the plot (typically the condition name).
    freqs : array
        Frequency axis in Hz, matching `means` and `band_mask` element-wise.
    means : array
        Grand average MSC values per frequency bin.
    plot_path : Path or None
        Path (including filename) to save the full-range plot to; the band-zoomed version
        is saved alongside it with '_zoomed' inserted before the extension. If None, the
        plot is created and closed without being saved.
    band_mask : array of bool, optional
        Boolean mask into `freqs`/`means` marking the DPOAE frequency band, used to shade
        the band and to determine the zoomed plot's x-limits. If None or all-False, no
        band is shaded, and saving (`plot_path` not None) will raise IndexError when
        computing the zoomed x-limits from the (then-empty) band frequencies.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    ax.plot(freqs, means, lw=0.9, color="#4C72B0", label='GA MSC')
    
    # shade the DPOAE band
    band_freqs = freqs[band_mask] if band_mask is not None else np.array([])
    if band_freqs.size:
        ax.axvspan(band_freqs[0], band_freqs[-1], alpha=0.15, color="orange",
                label="DPOAE band")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_title(f"{title}")
    #ax.set_ylim(bottom=min(0, y.min() - 0.01))
    ax.set_xlim(left=200, right=3000)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if plot_path is not None:
        plt.savefig(plot_path, dpi=1200)
        if band_freqs.size:
            ax.set_xlim(left=band_freqs[0]-300, right=band_freqs[-1]+300)
            plt.savefig(plot_path.with_name(plot_path.name.replace('.png', '_zoomed.png')), dpi=1200)
    plt.close()
    return 

def plot_GA_psd(title, freqs, means, plot_path, unit='dB'):
    """Plot a condition's grand average power spectral density (PSD) and save it to disk.

    Parameters
    ----------
    title : str
        Title displayed above the plot (typically the condition name).
    freqs : array
        Frequency axis in Hz, matching `means` element-wise.
    means : array
        Grand average PSD values per frequency bin.
    plot_path : Path or None
        Path (including filename) to save the plot to. If None, the plot is created and
        closed without being saved.
    unit : str, optional
        Unit label for the y-axis (e.g. 'dB' or 'Pa2_per_Hz'). Default is 'dB'.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    ax.plot(freqs, means, lw=0.9, color="#4C72B0", label='GA PSD')

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"PSD [{unit}]")
    ax.set_title(f"{title}")
    #ax.set_ylim(bottom=min(0, y.min() - 0.01))
    ax.set_xlim(left=250, right=3500)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if plot_path is not None:
        plt.savefig(plot_path, dpi=1200)
    plt.close()
    return 