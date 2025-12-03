"""
Figure for Reccomendation 2: Generate high signal-to-noise power spectra.
* Smoothing / transforming data
    a, Bandpass filter
    b, Interpolation
* Windowing and padding
    c, Windowing
    d, Padding
    e, Windowing + padding
* Computing spectra for short time windows
    f, Multitaper method
"""

# SET-UP #######################################################################

# imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import get_window

from specparam.utils.spectral import interpolate_spectrum
from neurodsp.sim import sim_combined
from neurodsp.utils import create_times
from neurodsp.spectral import compute_spectrum
from neurodsp.filt import filter_signal

import sys
sys.path.append('code')
from plt_utils import remove_spines, PANEL_FONTSIZE

# settings - figure
plt.style.use('mplstyle/nature_reviews.mplstyle')
FIGSIZE = [5, 7]

# settings - panel c-e
FS = 500 # sampling frequency
N_SECONDS = 2 # signal duration
PAD_FRACTION = 0.2 # pad duration / signal duration

# set random seed
np.random.seed(0)

# MAIN #########################################################################

def main():

    # create figure and gridspec
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    spec = gridspec.GridSpec(figure=fig, ncols=1, nrows=6, 
                             height_ratios=[0.02, 1, 0.02, 1.5, 0.02, 1.2])

    # plot panels a-e
    plot_panel_ab(fig, spec[1], fs=1200) 
    plot_panel_cde(fig, spec[3], fs=FS, n_seconds=N_SECONDS, 
                   pad_length=int(N_SECONDS*FS*PAD_FRACTION))
    
    # plot panel f: Cohen, 2014
    panel_c_path = "notebooks/images/cohen_2014_multitaper.png"
    ax_c = fig.add_subplot(spec[5])
    img = plt.imread(panel_c_path)[..., :3] # drop transparency
    img = (img - img.min(axis=(0,1))) / (img.max(axis=(0,1)) - img.min(axis=(0,1))) # sharpen image
    ax_c.imshow(img)
    ax_c.axis('off')

    # add panel titles
    titles = ["Smoothing / transforming data", 
              "Windowing and padding", 
              "Multitaper method"]
    for ii, title in enumerate(titles):
        ax_title = fig.add_subplot(spec[ii*2])
        ax_title.set_title(title, fontsize=12, pad=0)
        ax_title.axis("off")

    # add panel labels
    fig.text(0.01, 0.95, 'a', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.51, 0.95, 'b', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.67, 'c', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.51, 0.67, 'd', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.15, 0.45, 'e', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.29, 'f', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save/show
    fig.savefig(os.path.join('figures', 'figure_2.png'))


def plot_panel_ab(fig, subplot_spec, fs, n_seconds=10):
    # create nested subgridspec
    spec = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=subplot_spec, 
                                            width_ratios=[1, 1])
    ax_a = fig.add_subplot(spec[0, 0])
    ax_b = fig.add_subplot(spec[0, 1])

    # simulate signal (aperiodic activity + line noise)
    sim_components = {'sim_powerlaw': {'exponent' : -2},
                      'sim_oscillation': [{'freq' : 60}, {'freq' : 120}]}
    signal = sim_combined(n_seconds=n_seconds, fs=fs, 
                          components=sim_components, 
                          component_variances=[1, 0.5, 0.1])

    # apply bandstop filter
    signal_mfilt = signal.copy()
    for center_freq in [60, 120]:
        signal_mfilt = filter_signal(signal_mfilt, fs=fs, pass_type='bandstop',
                                     f_range=[center_freq-3, center_freq+3],
                                     filter_type='iir', butterworth_order=3)

    # compute power spectra
    freqs, psd = compute_spectrum(signal, fs=fs, method='welch')
    _, psd_mfilt = compute_spectrum(signal_mfilt, fs=fs, method='welch')

    # interpolate spectrum
    _, psd_interp = interpolate_spectrum(freqs, psd, [58, 62])
    _, psd_interp = interpolate_spectrum(freqs, psd_interp, [118, 122])

    # plot
    ax_a.loglog(freqs, psd, color='k', alpha=0.5, label='raw signal')
    ax_a.loglog(freqs, psd_mfilt, color='b', alpha=0.5, label='filtered signal')
    ax_b.loglog(freqs, psd, color='k', alpha=0.5, label='original psd')
    ax_b.loglog(freqs, psd_interp, color='b', alpha=0.5, label='interpolated psd')

    # label
    ax_a.set_title('Bandstop filter')
    ax_b.set_title('Interpolation')

    for ax in [ax_a, ax_b]:
        ax.set(xlabel='frequency (Hz)', ylabel='power (au)')
        ax.legend()

        # beautify
        remove_spines(ax)


def plot_panel_cde(fig, subplot_spec, fs, n_seconds, pad_length):
    """
    a, signal
    b, windowed signal
    c, padded signal
    d, padded windowed signal
    """

    # create nested subgridspec
    spec = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=subplot_spec, 
                                            height_ratios=[1, 1, 1, 6],
                                            width_ratios=[1, 1])
    ax_c_0 = fig.add_subplot(spec[0, 0])
    ax_c_1 = fig.add_subplot(spec[1, 0])
    ax_c_2 = fig.add_subplot(spec[2, 0])
    ax_d_0 = fig.add_subplot(spec[0, 1])
    ax_d_1 = fig.add_subplot(spec[1, 1])
    ax_d_2 = fig.add_subplot(spec[2, 1])

    # simulate signal (oscillation + aperiodic activity)
    sim_components = {'sim_powerlaw': {'exponent' : -2},
                      'sim_oscillation': [{'freq' : 10}, {'freq' : 45}]}
    signal = sim_combined(n_seconds=n_seconds, fs=fs, 
                          components=sim_components, 
                          component_variances=[1, 0.5, 0.2])
    time = create_times(n_seconds=n_seconds, fs=fs)
    ax_c_0.plot(time, signal, color='k')
    ax_d_0.plot(time, signal, color='k')

    # c, window signal
    window = get_window('hann', len(signal))
    signal_w = signal * window
    ax_c_1.plot(time, window, color='b')
    ax_c_2.plot(time, signal_w, color='k')

    # d, pad signal (mirror-padding and zero-padding)
    time_pad = create_times(n_seconds=(len(signal) / fs) + (pad_length * 2 / fs), 
                            fs=fs, start_val=-pad_length / fs)
    signal_pad_m = np.concatenate((np.flip(signal[:pad_length]), signal,
                                   np.flip(signal[-pad_length:])))
    signal_pad_z = np.concatenate((np.zeros(pad_length), signal,
                                   np.zeros(pad_length)))

    for signal_pad, ax in zip([signal_pad_m, signal_pad_z],
                              [ax_d_1, ax_d_2]):
        ax.plot(time_pad[pad_length-1:-pad_length-1], 
                signal_pad[pad_length-1:-pad_length-1], color='k')
        ax.plot(time_pad[:pad_length], signal_pad[:pad_length], color='b')
        ax.plot(time_pad[-pad_length:], signal_pad[-pad_length:], color='b')

    # e, window padded signal
    signal_pw = signal_pad_m * get_window('hann', len(signal_pad_m))
    spec_e = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec[3, :],
                                              width_ratios=[0.2, 1, 0.2])
    ax_e = fig.add_subplot(spec_e[0, 1])
    ax_e.plot(time_pad, signal_pw, color='k')

    # share axis
    for ax in [ax_c_1, ax_c_2]:
        ax.sharex(ax_c_0)
    for ax in [ax_d_1, ax_d_2]:
        ax.sharex(ax_d_0)

    # label
    for ax in [ax_c_2, ax_d_2, ax_e]:
        ax.set_xlabel('time (s)')
    for ax in [ax_c_1,  ax_d_1, ax_e]:
        ax.set_ylabel('voltage (au)')

    # remove clutter
    for ax in [ax_c_0, ax_c_1, ax_c_2, ax_d_0, ax_d_1, ax_d_2, ax_e]:
        ax.set_yticks([])
    for ax in [ax_c_0, ax_c_1, ax_d_0, ax_d_1]:
        ax.tick_params(axis='x', which='both', labelbottom=False)
    for ax in [ax_c_2, ax_d_2]:
        ax.tick_params(axis='x', which='both', labelbottom=True)

    # add titles
    ax_c_0.set_title('Signal')
    ax_c_1.set_title('Window (Hanning)')
    ax_c_2.set_title('Windowed signal')
    ax_d_0.set_title('Signal')
    ax_d_1.set_title('Padded signal (mirror)')
    ax_d_2.set_title('Padded signal (zero)')
    ax_e.set_title('Windowed padded signal')

    # beautify
    for ax in [ax_c_0, ax_c_1, ax_c_2, ax_d_0, ax_d_1, ax_d_2, ax_e]:
        remove_spines(ax)


if __name__ == "__main__":
    main()
