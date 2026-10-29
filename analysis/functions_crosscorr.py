"""Complex cross-correlation of a signal with the analytic signal of a template."""

import numpy as np
import scipy.signal as sig

def cross_corr(target, signal, normalize=True, limit=True, fs=44100):
    """Calculates the complex cross-correlation between the target and signal.

    The target is correlated with the real and the imaginary part of the analytic signal
    (Hilbert transform) of `signal`; the envelope is the magnitude of the complex result.

    Parameters
    ----------
    target : array 
        The first signal
    signal : array
        The second signal
    normalize : bool
        Whether the cross-correlation should be normalized. By default, this is set to True.
    limit : bool
        Whether the output should be limited to lags of +/- 1 second. By default, this is set to True.
    fs : int
        The sampling frequency in Hz, used for `limit`. Default is 44100.

    Returns
    -------
    lag : array
        The lag values of the cross-correlation in frames. 
    corr_real : array
        The real part of the cross-correlation.
    corr_imag : array
        The imaginary part of the cross-correlation.
    envelope : array
        The envelope of the cross-correlation.
    """
    signal_hilbert = sig.hilbert(signal)
    corr_real = sig.correlate(target, np.real(signal_hilbert), mode='full', method='auto') 
    corr_imag = sig.correlate(target, np.imag(signal_hilbert), mode='full', method='auto') 

    if normalize:
        corr_real = corr_real/ np.sqrt(np.sum(target**2) * np.sum(np.real(signal_hilbert)**2))
        corr_imag = corr_imag/ np.sqrt(np.sum(target**2) * np.sum(np.imag(signal_hilbert)**2))

    lag = np.arange(-len(target) + 1, len(target)) # in frames
    envelope = np.abs(corr_real + 1j * corr_imag)

    if limit:
        lag_idx = int(fs) # in order to take the correlation only for lags of max. 1 second
        length = len(lag)
        lag = lag[length//2 - lag_idx:length//2 + lag_idx + 1]
        corr_real = corr_real[length//2 - lag_idx:length//2 + lag_idx + 1]
        corr_imag = corr_imag[length//2 - lag_idx:length//2 + lag_idx + 1]
        envelope = envelope[length//2 - lag_idx:length//2 + lag_idx + 1]

    return lag, corr_real, corr_imag, envelope
