"""
Utility functions for spectrogram plotting and analysis.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_tfr(tfr, freqs, time, title=None, y_units="$\u00B5V^2/Hz$",
             fig=None, ax=None):
    """
    Plot time-frequency representation (TFR) of power (i.e. spectrogram).
    """
    
    # imports
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm

    # settings
    cmap = 'viridis'
    norm = LogNorm(vmin=np.min(tfr), vmax=np.max(tfr))

    # create figure
    if fig is None or ax is None:
        fig, ax = plt.subplots(1,1, figsize=[7.5, 3], constrained_layout=True)

    # plot
    ax.pcolormesh(time, freqs, tfr, cmap=cmap, norm=norm)
    ax.set(xlabel="time (s)", ylabel="frequnecy (Hz)")
    ax.axvline(0, color='k', linestyle='--')
    fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax, 
                 label=f"power ({y_units})")
    if title:
        ax.set_title(title)


def plot_evoked_tfr(tfr, freqs, time, title=None, annotate_time=0,
                    fig=None, ax=None):
    """
    Plot evoked spectrogram. Normalize power at each frequnecy to highlight
    event-related dynamics. Power values are z-scored across time, then
    the average baseline power is subtracted before plotting.
    """
    cmap = 'PiYG'#'coolwarm'

    # normalize power at each frequency by taking z-score across time
    tfr  = zscore_tfr(tfr)

    # subtract baseline power at each frequency
    tfr = subtract_baseline(tfr, time)

    # create figure
    if fig is None or ax is None:
        fig, ax = plt.subplots(1,1, figsize=[7.5, 3], constrained_layout=True)

    # plot normalized spectrogram
    ax.pcolormesh(time, freqs, tfr, cmap=cmap, shading='auto',
                  norm=TwoSlopeNorm(vcenter=0))
    ax.set(xlabel="time (s)", ylabel="frequnecy (Hz)")
    fig.colorbar(ax.pcolormesh(time, freqs, tfr, cmap=cmap, 
                               norm=TwoSlopeNorm(vcenter=0)), ax=ax,
                               label="normalized power (au)")
    
    # label
    if annotate_time:
        ax.axvline(annotate_time, color='k', linestyle='--')

    if title:
        ax.set_title(title)


def zscore_tfr(tfr):
    """
    Normalize time-frequency representation (TFR) by z-scoring each frequency.
    TFR should be 2D (frequency x time).

    Parameters
    ----------
    tfr : 2D array
        Time-frequency representation of power (spectrogram).

    Returns
    -------
    tfr_norm : 2D array
        Z-score normalized TFR.
    """

    # define z-score function
    def zscore(signal):
        return (signal - np.mean(signal)) / np.std(signal)
    
    # z-score normalize 
    tfr_norm = np.zeros_like(tfr)
    for i_freq in range(tfr.shape[0]):
        tfr_norm[i_freq] = zscore(tfr[i_freq])
        
    return tfr_norm


def subtract_baseline(signals, time, t_baseline=None):
    """
    Subtract baseline from signals. Baseline is defined as the mean of the
    signal between t_baseline[0] and t_baseline[1]. Signals should be 2D
    (signals x time).

    Parameters
    ----------
    signals : 2D array
        Signals to be baseline corrected.
    time : 1D array
        Time vector.
    t_baseline : 1D array, optional
        Time range for baseline (t_start, t_stop). If None, the baseline is
        defined as all time points before time 0.

    Returns
    -------
    signals_bl : 2D array
        Baseline corrected signals.
    """
    
    # create mask for baseline time range
    if t_baseline is None:
        t_baseline = (time[0], 0)
    mask_bl = ((time>t_baseline[0]) & (time<t_baseline[1]))
    
    # subtract baseline from each signal
    signals_bl = np.zeros_like(signals)
    for ii in range(len(signals)):
        signals_bl[ii] = signals[ii] - np.mean(signals[ii, mask_bl])
    
    return signals_bl


def crop_tfr(tfr, time, time_range):
    """
    Crop time-frequency representation (TFR) to a specified time range.
    TFR can be mulitdimensional (time must be last dimension).

    Parameters
    ----------
    tfr : array
        Time-frequency representation of power (spectrogram).
    time : 1D array
        Associated time vector (length should be equal to that of
        the last dimension of tfr).
    time_range : 1D array
        Time range to crop (t_start, t_stop).

    Returns
    -------
    tfr, time : array, array
        Cropped TFR and time vector.
    """
    
    tfr = tfr[..., (time>time_range[0]) & (time<=time_range[1])]
    time = time[(time>time_range[0]) & (time<=time_range[1])]
    
    return tfr, time
