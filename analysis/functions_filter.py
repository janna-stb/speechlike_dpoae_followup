"""Butterworth filtering (zero-phase)."""

import scipy.signal as sig

def butter_filter(x, cutoff, btype, output='sos', order=3, fs=44100):
    """Applies a Butterworth filter to the signal.

    Parameters
    ----------
    x : array
        The signal to be filtered
    cutoff : float
        The cutoff frequency. If btype is 'bandpass' or 'bandstop', cutoff should be a list with two values.
    btype : str
        The type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
    output : str
        Filter representation used for filtering, 'ba' or 'sos' (default, numerically more stable)
    order : int
        The order of the filter. Default is 3; because the filter is applied forwards and
        backwards (zero phase), the effective order is twice as high.
    fs : int
        The sampling frequency

    Returns
    -------
    x_filt : array
        The filtered signal
    """

    if output == 'ba':
        b, a = sig.butter(order, cutoff, btype=btype, output='ba', fs=fs)
        x_filt = sig.filtfilt(b, a, x)
    elif output == 'sos':
        sos = sig.butter(order, cutoff, btype=btype, output='sos', fs=fs)
        x_filt = sig.sosfiltfilt(sos, x)
    
    return x_filt
