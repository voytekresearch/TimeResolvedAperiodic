"""
Figure 1: Time-resolved Parameterization

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

from neurodsp.sim import (
    sim_synaptic_current,
    sim_knee
)
from neurodsp.spectral import compute_spectrum

from neurodsp.utils import create_times
from neurodsp.sim.utils import rotate_timeseries
from specparam import SpectralModel, SpectralTimeModel

import fooof
from fooof.utils.params import compute_knee_frequency

import sys
sys.path.append('code')
from plt_utils import remove_spines, FIGURE_WIDTH, PANEL_FONTSIZE
from tfr_utils import plot_evoked_tfr

# settings - figure
plt.style.use('mplstyle/nature_reviews.mplstyle')
FIGSIZE = [FIGURE_WIDTH+2, 10]
TIME_POINTS = [-0.35, -0.25, -0.15, 1.35] # which to plot
COLORS = sns.color_palette("Blues", len(TIME_POINTS))
TITLE_FONTSIZE = PANEL_FONTSIZE - 5
# sns.set_context('talk')

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
    gs = gridspec.GridSpec(figure=fig, ncols=1, nrows=5, 
                           height_ratios=[0.75, 0.5, 0.5, 0.5, 0.75])

    # Add variable freq range plots
    ax_e = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0],
                                            width_ratios=[1, 1, 1, 1])
    plot_variable_freq_ranges(fig, ax_e)
    plot_diff_time_wins(fig, plt.subplot(ax_e[3]))

    # Simulate and plot bursty oscillation
    ax_a = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1],
                                            width_ratios=[1])
    sig, _ = sim_and_plot_signal(plt.subplot(ax_a[0]))

    # Compute and plot TFR
    ax_b = fig.add_subplot(gs[2])
    tfr, time_tfr, freqs = compute_and_plot_tfr(sig, fig, ax_b)

    # Plot spectral parameterization
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[3],
                                            width_ratios=[1, 1, 1, 0.001, 1])
    ax_c_0 = fig.add_subplot(gs_c[0])
    ax_c_1 = fig.add_subplot(gs_c[1])
    ax_c_2 = fig.add_subplot(gs_c[2])
    ax_c_3 = fig.add_subplot(gs_c[4])
    axes_c = [ax_c_0, ax_c_1, ax_c_2, ax_c_3]
    for ax, tp, col in zip(axes_c, TIME_POINTS, COLORS):
        plot_sparam_psd(tfr, time_tfr, freqs, ax=ax, tp=tp)
    ax_c_1.set_title("                                Parameterization of spectrogram bins", fontsize=TITLE_FONTSIZE)
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
    ax_d = fig.add_subplot(gs[4])
    ax_d.set_title("Time-resolved spectral features", fontsize=TITLE_FONTSIZE)
    compute_and_plot_sliding_window_params(tfr, time_tfr, freqs, ax=ax_d)

    # add panel labels
    fig.text(0.01, 0.97, 'A', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.75, 0.97, 'B', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.78, 'C', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.62, 'D', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.43, 'E', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.28, 'F', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # remove spines
    for ax in [ ax_b, *axes_c, ax_d]:
        remove_spines(ax)

    # # Save
    fig.savefig('figures\\figure_0.png')#os.path.join('figures', 'figure_0.png'))


def sim_and_plot_signal(ax):
    
    # simulate neural time-series
    white = np.random.randn(N_SECONDS*FS)
    sig_0 = rotate_timeseries(white, FS, -EXPONENT, F_ROTATION)
    sig_1 = rotate_timeseries(sig_0, FS, DELTA_EXP, F_ROTATION)
    sig_0 = sig_0[:int(FS*-T_MIN)]
    sig_1 = sig_1[int(FS*-T_MIN):]
    sig_0 = sig_0 - np.mean(sig_0)
    sig_1 = sig_1 - np.mean(sig_1)
    signal = np.concatenate((-sig_0, sig_1))
    time = create_times(N_SECONDS, FS, start_val=T_MIN)

    # Plot the simulated data, in the time domain
    ax.plot(time, signal, color='k', linewidth=1)
    ax.set(xlabel="time (s)", ylabel="voltage (au)")
    ax.set_title("Simulated neural time-series", fontsize=TITLE_FONTSIZE)
    ax.set_xlim(T_MIN, N_SECONDS+T_MIN)
    ax.set_xlabel(ax.get_xlabel(), fontsize=TITLE_FONTSIZE, weight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=TITLE_FONTSIZE, weight='bold')


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
    ax.set_title('Spectrogram', fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(ax.get_xlabel(), fontsize=TITLE_FONTSIZE, weight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=TITLE_FONTSIZE, weight='bold')

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
    ax.loglog(fm.data.freqs, 10**fm.results.model._ap_fit, color="b", 
              label="aperiodic fit", linestyle='--')
    # ax.set(xlabel="frequency (Hz)", ylabel="power (au)") 
    ax.set_xlabel("frequency (Hz)", fontsize=TITLE_FONTSIZE, weight='bold')
    ax.set_ylabel("power (au)", fontsize=TITLE_FONTSIZE, weight='bold')
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
    ax.plot(time, exponent, color='b', label='exponent')
    ax.legend(loc='lower left', fontsize=PANEL_FONTSIZE)

    offset = stm.get_params('aperiodic', 'offset')
    ax2 = ax.twinx()
    ax2.plot(time, offset, color='b', linestyle='--', label='offset')
    ax2.legend(loc='lower right', fontsize=PANEL_FONTSIZE)

    # Label
    ax.set_xlabel("time (s)", fontsize=TITLE_FONTSIZE, weight='bold')
    ax.set_ylabel("exponent", fontsize=TITLE_FONTSIZE, weight='bold')
    ax2.set_ylabel("offset", fontsize=TITLE_FONTSIZE, weight='bold')
    add_task_labels(ax)
    # plt.legend(['exponent','offset'])


def plot_variable_freq_ranges(fig, ax_in): 

    knees = [1,5,10,15,20,25,30][::-1]
    seeds = np.arange(40,40+len(knees))
    # tau_E = (0.005, (1 / (2 * np.pi * highKnee)))  # rise, decay
    # tau_I = (0.005, (1 / (2 * np.pi * lowKnee)))  # rise, decay
    n_E = 8000
    n_I = 2000
    firRate_E = 2  # Hz
    firRate_I = 2  # Hz
    scale_I = 1
    n_seconds = 100
    fs = 1000
    f_range = (1, 150)
    colors = sns.color_palette('crest', n_colors=len(knees))
    alpha_min = 0.45
    alpha_max = 1
    alphas = np.linspace(alpha_min, alpha_max, len(knees)) #[1,0.95,0.85,0.75,0.7][::-1]


    for i,knee in enumerate(knees):
        
        tau = (0.005, (1 / (2 * np.pi * knee)))  # rise, decay

        np.random.seed(seeds[i])#42)
        ## Use these to simulate currents from both of these; this will give us a slower (inhibitory) signal, and a faster (excitatory) one
        sig = sim_synaptic_current(
            n_seconds=n_seconds,
            fs=fs,
            n_neurons=n_E,
            firing_rate=firRate_E,
            tau_r=tau[0],
            tau_d=tau[1],
        )
        # sig = sim_knee(n_seconds=n_seconds, fs=fs, exponent1=-0.1,exponent2=-2, knee=knee)
        time = np.arange(0, len(sig)) / fs


        ## Calc PSDs for each timeseries and plot; these should have different knees
        freqs_slow, powers_slow = compute_spectrum(
            sig, fs, f_range=f_range, avg_type="mean", nperseg=fs, noverlap=fs / 2
        )

        ax = plt.subplot(ax_in[0])
        ax.loglog(freqs_slow, (powers_slow/powers_slow[0]), label="Inhibitory", color=colors[i], zorder=0)
        # ax.axvline(knee, color=colors[i], linewidth=1.5)
        ax.scatter(knee, (powers_slow/powers_slow[0])[np.where(freqs_slow == knee)[0]], color=colors[i], edgecolors='k', zorder=1)
        ax.set_xlabel('frequency (log Hz)', fontsize=TITLE_FONTSIZE, weight='bold')
        ax.set_ylabel('\npower (log)', fontsize=TITLE_FONTSIZE, weight='bold')
        # ax.set_facecolor('lightgrey')
        # plt.legend()
        
        init_settings = {
            "peak_width_limits": (2, 14),
            "peak_threshold": 2,
            "max_n_peaks": 0,
        }

        init_settings["aperiodic_mode"] = "fixed"
        fm_gt = fooof.FOOOF(**init_settings)
        fm_gt.fit(freqs_slow, power_spectrum=powers_slow, freq_range = (knee, 150))
        # fm_narrow.plot()
        exp_gt= fm_gt.get_params('aperiodic_params','exponent')

        init_settings["aperiodic_mode"] = "knee"
        fm_narrow = fooof.FOOOF(**init_settings)
        fm_narrow.fit(freqs_slow, power_spectrum=powers_slow, freq_range = (20, 100))
        # fm_narrow.plot()
        exp_narrow = fm_narrow.get_params('aperiodic_params','exponent')
        # print(exp_narrow)
        knee_narrow = compute_knee_frequency(fm_narrow.get_params('aperiodic_params','knee'), exp_narrow)
        # print(knee)
        # print('.')

        init_settings["aperiodic_mode"] = "knee"
        fm_wide = fooof.FOOOF(**init_settings)
        fm_wide.fit(freqs_slow, power_spectrum=powers_slow, freq_range = (0, 100))
        # fm_wide.plot()
        exp_wide = fm_wide.get_params('aperiodic_params','exponent')
        # print(exp_wide)
        knee_wide = compute_knee_frequency(fm_wide.get_params('aperiodic_params','knee'), exp_wide)
        # print(knee)
        # print('.................')

        fm_low = fooof.FOOOF(**init_settings)
        fm_low.fit(freqs_slow, power_spectrum=powers_slow, freq_range = (0, 20))
        # fm_wide.plot()
        exp_low = fm_low.get_params('aperiodic_params','exponent')
        # print(exp_wide)
        knee_low = compute_knee_frequency(fm_low.get_params('aperiodic_params','knee'), exp_wide)

        ax = plt.subplot(ax_in[1])
        scatter_std = 0.025
        ax.set_xlim(-0.25,2.5)
        ax.scatter(np.random.normal(0,scatter_std), exp_low, color=colors[i])
        ax.scatter(np.random.normal(0.75,scatter_std), exp_narrow, color=colors[i])
        ax.scatter(np.random.normal(1.5,scatter_std), exp_wide, color=colors[i])
        ax.scatter(np.random.normal(2.25,scatter_std), exp_gt, color=colors[i])
        ax.set_xticks([0,0.75, 1.5,2.25])
        ax.set_xticklabels(['low \n(0-20)','high \n(20-100)', 'broadband \n(0-100)', 'ground \ntruth'], fontsize='medium')
        ax.set_xlabel('fit range (Hz)', fontsize=TITLE_FONTSIZE, weight='bold')
        ax.set_ylabel('exponent estimate', fontsize=TITLE_FONTSIZE, weight='bold')
        # ax.set_facecolor('lightgrey')

        ax = plt.subplot(ax_in[2])
        ax.set_xlim(-0.25,2.5)
        ax.scatter(np.random.normal(0,scatter_std), knee_low, color=colors[i])
        ax.scatter(np.random.normal(0.75,scatter_std), knee_narrow, color=colors[i])
        ax.scatter(np.random.normal(1.5,scatter_std), knee_wide, color=colors[i])
        ax.scatter(np.random.normal(2.25,scatter_std), knee, color=colors[i])
        ax.set_xticks([0,0.75, 1.5,2.25])
        ax.set_xticklabels(['low \n(0-20)','high \n(20-100)', 'broadband \n(0-100)', 'ground \ntruth'], fontsize='medium')
        ax.set_xlabel('fit range (Hz)', fontsize=TITLE_FONTSIZE, weight='bold')
        ax.set_ylabel('knee estimate', fontsize=TITLE_FONTSIZE, weight='bold')
        # ax.set_facecolor('lightgrey')


def plot_diff_time_wins(fig, ax):
    lw = 2

    cols = sns.dark_palette('#69d', n_colors=3)
    # Set some general settings, to be used across all simulations
    fs = 500
    n_seconds = 50
    # Create a times vector for the simulations
    times = create_times(n_seconds, fs)
    np.random.seed(35)
    # Simulate another knee signal, with different exponents & knee
    knee_ap2 = sim_knee(n_seconds, fs, exponent1=-0.5, exponent2=-2, knee=2)

    freqs, knee_psd1 = compute_spectrum(knee_ap2[times<1], fs)
    freqs, knee_psd2 = compute_spectrum(knee_ap2, fs)
    freqs, knee_psd3 = compute_spectrum(knee_ap2[times<10], fs)

    freqs_mask = (freqs>=1) & (freqs<50)
    knee_psd1, knee_psd2, knee_psd3 = knee_psd1[freqs_mask], knee_psd2[freqs_mask], knee_psd3[freqs_mask]
    freqs = freqs[freqs_mask]

    # Plot the simulated data, in the frequency domain
    ax.loglog(freqs, (knee_psd1), label = 'Short Time Window (1s)', color=cols[0], linewidth=lw, alpha=1)
    ax.loglog(freqs, (knee_psd3), label = 'Medium Time Window (10s)', color=cols[1], linewidth=lw)
    ax.loglog(freqs, (knee_psd2), label='Long Time Window (50s)', color=cols[2], linewidth=lw, alpha=0.85)
    ax.set_xlabel('frequency (log Hz)', fontsize=TITLE_FONTSIZE, weight='bold')
    ax.set_ylabel('\npower (log)', fontsize=TITLE_FONTSIZE, weight='bold')
    ax.set_xlim(-5,20)#freqs[-1]+20)
    plt.axvline(2, color='grey')
    # ax.set_facecolor('darkgrey')
    plt.legend()

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
