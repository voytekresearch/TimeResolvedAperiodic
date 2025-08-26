"""
Figure 0: Time-resolved Parameterization

This is a conceptual figure, to introduce the methodology of time-resolved 
parameterization.

Panels:
a, Simulated neural time-series
b, Time-frequency representation
c, Parameterization of TFR bins (1, 2... N)
d, Time-resolved spectral features

"""


# SET-UP #######################################################################

# imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mne.time_frequency import tfr_array_multitaper
import seaborn as sns

from neurodsp.utils import create_times
from neurodsp.sim.utils import rotate_timeseries
from specparam import SpectralModel, SpectralTimeModel

import sys
sys.path.append('code')
from plt_utils import remove_spines, FIGURE_WIDTH, PANEL_FONTSIZE
from tfr_utils import plot_evoked_tfr

# settings - figure
plt.style.use('mplstyle/nature_reviews.mplstyle')
FIGSIZE = [FIGURE_WIDTH, 7]
TIME_POINTS = [-0.35, -0.25, -0.15, 1.35] # which to plot
COLORS = sns.color_palette("Greens", len(TIME_POINTS))

# settings - simulation parameters
N_SECONDS = 2 # signal duration (s)
T_MIN = -0.5 # start time (s)
FS = 1000 # sampling frequency (Hz)
EXPONENT = -2.5 # baseline exponent
DELTA_EXP = -1 # task-evoked change in exponent (negative for flattening)
F_ROTATION = 45 # rotation frequency (Hz)

# settings - fitting parameters
SPECPARAM_SETTINGS = {
    'aperiodic_mode' : 'fixed',
    'max_n_peaks' : 0,
    'verbose' : False,
}

# settings - multitaper
TFR_WINDOW = 0.3 # window length (s)
FREQ_BANDWIDTH = 7 # frequency bandwidth (Hz)

# set random seed
np.random.seed(39)

# MAIN #########################################################################

def main():

    # create figure and gridspec
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = gridspec.GridSpec(figure=fig, ncols=1, nrows=4, 
                           height_ratios=[0.5, 0.75, 0.75, 0.5])

    # Simulate and plot bursty oscillation
    ax_a = fig.add_subplot(gs[0])
    sig, _ = sim_and_plot_signal(ax_a)

    # Compute and plot TFR
    ax_b = fig.add_subplot(gs[1])
    tfr, time_tfr, freqs = compute_and_plot_tfr(sig, fig, ax_b)

    # Plot spectral parameterization
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[2],
                                            width_ratios=[1, 1, 1, 0.001, 1])
    ax_c_0 = fig.add_subplot(gs_c[0])
    ax_c_1 = fig.add_subplot(gs_c[1])
    ax_c_2 = fig.add_subplot(gs_c[2])
    ax_c_3 = fig.add_subplot(gs_c[4])
    axes_c = [ax_c_0, ax_c_1, ax_c_2, ax_c_3]
    for ax, tp, col in zip(axes_c, TIME_POINTS, COLORS):
        plot_sparam_psd(tfr, time_tfr, freqs, ax=ax, tp=tp)
    ax_c_1.set_title("                                Parameterization of spectrogram bins")
    for ax in [ax_c_1, ax_c_2, ax_c_3]:
        ax.sharey(ax_c_0)
        ax.label_outer()
    for ax, col in zip(axes_c, COLORS):
        add_background(ax, col)

    # add large text elipsis
    ax_c_x = fig.add_subplot(gs_c[3])
    ax_c_x.text(0.5, 0.5, r"$\cdots$", fontsize=40, ha="center", va="center")
    ax_c_x.axis("off")

    # Compute and plot sliding window parameters
    ax_d = fig.add_subplot(gs[3, :])
    ax_d.set_title("Time-resolved spectral features")
    compute_and_plot_sliding_window_params(tfr, time_tfr, freqs, ax=ax_d)

    # add panel labels
    fig.text(0.01, 0.97, 'a', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.76, 'b', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.43, 'c', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.21, 'd', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # remove spines
    for ax in [ax_a, ax_b, *axes_c, ax_d]:
        remove_spines(ax)

    # # Save
    fig.savefig(os.path.join('figures', 'figure_0.png'))


def sim_and_plot_signal(ax):
    
    # simulate neural time-series
    white = np.random.randn(N_SECONDS*FS)
    sig_0 = rotate_timeseries(white, FS, -EXPONENT, F_ROTATION)
    sig_1 = rotate_timeseries(sig_0, FS, DELTA_EXP, F_ROTATION)
    sig_0 = sig_0[:int(FS*-T_MIN)]
    sig_1 = sig_1[int(FS*-T_MIN):]
    sig_0 = sig_0 - np.mean(sig_0)
    sig_1 = sig_1 - np.mean(sig_1)
    signal = np.concatenate((sig_0, sig_1))
    time = create_times(N_SECONDS, FS, start_val=T_MIN)

    # Plot the simulated data, in the time domain
    ax.plot(time, signal, color='k', linewidth=1)
    ax.set(xlabel="time (s)", ylabel="voltage (au)", 
           title="Simulated neural time-series")
    ax.set_xlim(T_MIN, N_SECONDS+T_MIN)

    # label task periods
    add_task_labels(ax)

    return signal, time


def compute_and_plot_tfr(sig, fig, ax):

    # Compute PSD using the multitaper method
    freqs = np.linspace(1, 100, 100)
    n_cycles = freqs * TFR_WINDOW # set n_cycles based on fixed time window length
    time_bandwidth =  TFR_WINDOW * FREQ_BANDWIDTH # must be >= 2
    tfr = tfr_array_multitaper(sig[np.newaxis, np.newaxis, :], sfreq=FS, 
                               freqs=freqs, n_cycles=n_cycles, decim=10,
                               time_bandwidth=time_bandwidth, output="power")

    # Extract TFR (squeeze unnecessary dimensions)
    tfr_power = tfr.squeeze()  # Shape becomes (n_frequencies, n_times)
    time_tfr = create_times(N_SECONDS, FS/10, start_val=T_MIN)

    # Plot the TFR
    plot_evoked_tfr(tfr_power, freqs, time_tfr, title="Spectrogram", fig=fig, 
                    ax=ax, annotate_time=None)

    # plot boxes around TIME_POINTS ranges
    for tp, color in zip(TIME_POINTS, COLORS):
        start = tp - TFR_WINDOW / 2 * .95
        end   = tp + TFR_WINDOW / 2 * .95
        width = end - start
        
        rect = patches.Rectangle(
            (start, ax.get_ylim()[0]-1.5), #*1.5),
            width,
            (ax.get_ylim()[1] - ax.get_ylim()[0]) + 3, #.98,
            linewidth=3,
            edgecolor=color,
            facecolor='none',
            alpha=1,
            clip_on=False
        )
        ax.add_patch(rect)

    return tfr, time_tfr, freqs


def plot_sparam_psd(tfr, time, freqs, tp, ax):
    """Plot spectral parameterization."""

    # Compute spectral parameterization
    tp_idx = np.argmin(np.abs(time - tp))
    powers = tfr[0, 0, :, tp_idx]
    fm = SpectralModel(**SPECPARAM_SETTINGS)
    fm.fit(freqs, powers)

    # Plot PSD and aperiodic fit
    ax.loglog(freqs, powers, color="k", label="spectrum")
    ax.loglog(fm.freqs, 10**fm._ap_fit, color="b", 
              label="aperiodic fit", linestyle='--')
    ax.set(xlabel="frequency (Hz)", ylabel="power (au)") 
    ax.legend(loc="lower left")

    # label time
    ax.text(0.5, 0.93, f"time: {tp:.2f} s", ha='center', va='center', 
            transform=ax.transAxes)


def compute_and_plot_sliding_window_params(tfr, time, freqs, ax):
    """Compute and plot sliding window parameters."""
    # parameterize
    stm = SpectralTimeModel(**SPECPARAM_SETTINGS)
    stm.fit(freqs, np.squeeze(tfr))

    # plot exponent
    exponent = stm.get_params('aperiodic', 'exponent')
    ax.plot(time, exponent, color='b')

    # Label
    ax.set_xlabel("time (s)")
    ax.set_ylabel("exponent")
    add_task_labels(ax)


def add_background(ax, background_color):

    # Plot background color
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    rect = plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        1.3 * y_max - y_min,
        color=background_color,
        alpha=0.5,
        zorder=-1,
    )
    ax.add_patch(rect)


def add_task_labels(ax):
    # label task periods
    ax.axvspan(0, 1.0, color='y', alpha=0.2)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.1)
    ax.text(-.25, ax.get_ylim()[1]*0.98, 'baseline', ha='center', va='top')
    ax.text(0.50, ax.get_ylim()[1]*0.98, 'encoding', ha='center', va='top')
    ax.text(1.25, ax.get_ylim()[1]*0.98, 'delay', ha='center', va='top')


if __name__ == "__main__":
    main()
