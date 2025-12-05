"""
Figure 0: Background / Pedgogical Fig

This is a conceptual figure, to introduce the methodology of time-resolved 
parameterization.

Panels:
a, Simulated neural time-series
b, 

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
    sim_knee,
    sim_combined,
    sim_oscillation
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
FIGSIZE = [FIGURE_WIDTH+2, 17]
# TIME_POINTS = [-0.35, -0.25, -0.15, 1.35] # which to plot
# COLORS = sns.color_palette("Greens", len(TIME_POINTS))
TITLE_FONTSIZE = PANEL_FONTSIZE - 5
sns.set_context('talk')

# settings - simulation parameters
N_SECONDS = 5 # signal duration (s)
FS = 1000 # sampling frequency (Hz)
EXPONENT = -1.5 # baseline exponent
DELTA_EXP = 1.5 # task-evoked change in exponent (negative for flattening, pos for steepening)
F_ROTATION = 35 # rotation frequency (Hz)
OSC_FREQ = 20 # freq of osc rhythm
DELTA_OSC_AMP = 1.20 # how much we'll increase the osc amp for the post-stim signal
TIMESERIES_YLIM = (-25, 25) # yaxis lim for timeseries plot
BAR_YLIM = (0,100) # yaxis bounds for barplot
PSD_YLIM = (-0.16525260077205794, 121.47836693568351)# yaxis bounds for PSD plot

# settings - fitting parameters
SPECPARAM_SETTINGS = {
    'aperiodic_mode' : 'fixed',
    'max_n_peaks' : 1,
    'verbose' : False,
}

# settings - multitaper
TFR_WINDOW = 0.3 # window length (s)
FREQ_BANDWIDTH = 7 # frequency bandwidth (Hz)

# set random seed
np.random.seed(42)
colors_pal = (sns.color_palette('crest'))
prestim_color = colors_pal[0]
poststim_color = colors_pal[3]

# MAIN #########################################################################

def main():

    # create figure and gridspec
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = gridspec.GridSpec(figure=fig, ncols=1, nrows=5, 
                           height_ratios=[0.5,1, 0.5, 0.5, 0.1])

    # # Add variable freq range plots
    # ax_e = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0],
    #                                         width_ratios=[1, 1, 1, 1])
    # plot_variable_freq_ranges(fig, ax_e)
    # plot_diff_time_wins(fig, plt.subplot(ax_e[3]))

    # Simulate and plot aperiodic + oscillation with events
    ax_top = gridspec.GridSpecFromSubplotSpec(ncols=3, nrows=1, subplot_spec=gs[0], width_ratios=[2,5,2])
    events = [2.5]#[0.75, 1.25, 3.25, 4.5]
    event_win = 1
    sig_no, times_no = generate_modulated_signal(events, event_win, FS, rotate_aper=1)

    ax_no = fig.add_subplot(ax_top[1])
    for ev in events: 
        ax_no.axvline(ev, color='grey', linewidth=3)
        ax_no.axvspan(xmin = ev-event_win, xmax=ev,  color = prestim_color)
        ax_no.axvspan(xmin = ev, xmax=ev+event_win, color = poststim_color)
    # plot the timeseries    
    ax_no.plot(times_no, sig_no, color='k', alpha=0.85)
    ax_no.set_ylim(TIMESERIES_YLIM)

    ax_no.set_xlabel('Time (s)')
    ax_no.set_ylabel('Voltage (au)')

    # Plot PSDs
    ax_mid = gridspec.GridSpecFromSubplotSpec(ncols=2, nrows=1, subplot_spec=gs[2],
                                            width_ratios=[1,1])
    ax_psd_no = fig.add_subplot(ax_mid[0])
    ax_psd_rot = fig.add_subplot(ax_mid[1])

    # Plot barplots of spectral params
    ax_bottom = gridspec.GridSpecFromSubplotSpec(ncols=2, nrows=1, subplot_spec=gs[3],
                                            width_ratios=[1,1])
    ax_bar_no = fig.add_subplot(ax_bottom[0])
    ax_bar_rot = fig.add_subplot(ax_bottom[1])

    axes_list = [(ax_psd_no, ax_bar_no, sig_no, 1),(ax_psd_rot, ax_bar_rot, sig_no, 0)]
    ev = events[0]
    ev_idx = int(ev*FS)
    for ax_a, ax_b, signal, tb in axes_list:
        axs = (ax_a, ax_b)
        plot_prestim_poststim_psd(signal, ev_idx, event_win, FS, axs, total_bandpow = tb)
    ax_psd_rot.set_ylim(ax_psd_no.get_ylim())

    ax_psd_no.set_xlabel('Freq (Hz))')
    ax_psd_no.set_ylabel('Log Power (au)')
    ax_psd_rot.set_xlabel('Freq (Hz)')
    ax_psd_rot.set_ylabel('Log Power (au)')

    # add panel labels
    # fig.text(0.01, 0.97, 'A', fontsize=PANEL_FONTSIZE, fontweight='bold')
    # fig.text(0.75, 0.97, 'B', fontsize=PANEL_FONTSIZE, fontweight='bold')
    # fig.text(0.01, 0.76, 'C', fontsize=PANEL_FONTSIZE, fontweight='bold')
    # fig.text(0.01, 0.60, 'D', fontsize=PANEL_FONTSIZE, fontweight='bold')
    # fig.text(0.01, 0.43, 'E', fontsize=PANEL_FONTSIZE, fontweight='bold')
    # fig.text(0.01, 0.25, 'F', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # remove spines
    # for ax in [ ax_b, *axes_c, ax_d]:
    #     remove_spines(ax)

    # # Save
    fig.savefig('figures\\figure_pedagogical.png')#os.path.join('figures', 'figure_0.png'))

def generate_modulated_signal(events, event_win, fs, rotate_aper):

    sim_components = {
        "sim_powerlaw": {"exponent": EXPONENT}
    }
    sig = sim_combined(n_seconds=N_SECONDS, fs=fs, components=sim_components)
    osc_sig = sim_oscillation(n_seconds=N_SECONDS, fs=fs, freq=OSC_FREQ)

    # get prestim signal as aper + osc
    sig = (sig*2) + (osc_sig)# crank up the ratio of aper sig 

    # get alpha sig which we'll add to get the poststim win
    osc_sig_mod = (osc_sig*DELTA_OSC_AMP)

    times = create_times(n_seconds=N_SECONDS, fs=fs)

    for ev in events:
        ev_idx = int(ev*fs)
        ev_idx_end = int(ev_idx + (event_win*fs))
        print(ev_idx, ev_idx_end)

        mod_sig = sig[ev_idx : ev_idx_end]
        osc_sig_add = osc_sig_mod[ev_idx : ev_idx_end]
        if rotate_aper:
            rotated = rotate_timeseries(sig=mod_sig, fs=fs, delta_exp=DELTA_EXP, f_rotation=F_ROTATION)
        else:
            rotated = mod_sig
        rotated = rotated + osc_sig_add
        sig[ev_idx : ev_idx_end ] = rotated

    return sig, times

def plot_prestim_poststim_psd(sig, ev_idx, event_win, fs, axs, total_bandpow = 0):

    psd_ax, bar_ax = axs

    freqs, pows_pre = compute_spectrum(sig = sig[ev_idx - int(event_win*fs) : ev_idx ], fs=fs)
    freqs, pows_post = compute_spectrum(sig = sig[ev_idx : ev_idx + int(event_win*fs) ], fs=fs)
    pows_pre = pows_pre[(freqs > 0.5) & (freqs < 50)]
    pows_post = pows_post[(freqs > 0.5) & (freqs < 50)]
    freqs = freqs[(freqs > 0.5) & (freqs < 50)]

    alpha_mask = (freqs > 18) & (freqs <=22) #changed to beta
    pre_total_power = np.mean(pows_pre[alpha_mask])
    pst_total_power = np.mean(pows_post[alpha_mask])
    delta_total = ((pst_total_power - pre_total_power) / pre_total_power)*100

    specpar_pre = fooof.FOOOF(**SPECPARAM_SETTINGS)
    specpar_pre.fit(freqs=freqs, power_spectrum=pows_pre, freq_range=(0.5, 50))    
    specpar_pst = fooof.FOOOF(**SPECPARAM_SETTINGS)
    specpar_pst.fit(freqs=freqs, power_spectrum=pows_post, freq_range=(0.5, 50))

    # put periodic and aper back into linear space
    pre_ap_fit = 10**(specpar_pre._ap_fit)
    pst_ap_fit = 10**(specpar_pst._ap_fit)
    # pre_per_fit = 10**(specpar_pre._spectrum_flat)
    # pst_per_fit = 10**(specpar_pst._spectrum_flat)

    # calc % change pre to post stim
    pre_pk_amp = specpar_pre.get_results().peak_params[0][1]
    pst_pk_amp = specpar_pst.get_results().peak_params[0][1]
    flat_spec_delta = ((pst_pk_amp - pre_pk_amp) / pre_pk_amp)*100

    pre_aper_exp = specpar_pre.get_results().aperiodic_params[1]
    pst_aper_exp = specpar_pst.get_results().aperiodic_params[1]
    ap_delta = ((pst_aper_exp - pre_aper_exp) / pre_aper_exp)*100

    # flat_spec_delta = (((pst_per_fit) - (pre_per_fit)) / pre_per_fit)*100 # diff of periodic sig, normalized
    # flat_spec_delta = np.mean(flat_spec_delta[alpha_mask])
    # ap_delta = (((pst_ap_fit) - (pre_ap_fit)) / pre_ap_fit)*100
    # ap_delta = np.mean(ap_delta[alpha_mask])

    psd_ax.plot(freqs,pows_pre, color = prestim_color, alpha=0.85, linewidth=3)
    psd_ax.plot(freqs,pows_post, color = poststim_color, alpha=0.85, linewidth=3)

    # psd_ax.axvspan(xmin=8, xmax=12, color='grey', alpha=0.5)

    # psd_ax.set_xscale('log')
    # psd_ax.set_ylim(PSD_YLIM)

    if total_bandpow:
        bar_ax.bar([u'Δ oscillation', u'Δ aperiodic'], height=[ delta_total, 0], color='grey')
        # psd_ax.plot(freqs,pre_ap_fit, color = prestim_color, linewidth=1)
        # psd_ax.plot(freqs,pst_ap_fit, color = poststim_color, linewidth=1)
        
        psd_ax.plot(OSC_FREQ, pows_pre[(freqs == OSC_FREQ)][0], color=prestim_color, marker='o', alpha=0.8)
        psd_ax.plot(OSC_FREQ, pows_post[(freqs == OSC_FREQ)][0], color=poststim_color, marker='o', alpha=0.8)

        # psd_ax.axvspan(xmin=OSC_FREQ-2, xmax=OSC_FREQ+2, color=prestim_color, alpha=0.5, ymax = pows_pre[(freqs == OSC_FREQ)][0])
        # psd_ax.axvspan(xmin=OSC_FREQ-2, xmax=OSC_FREQ+2, color=poststim_color, alpha=0.5, ymax = pows_post[(freqs == OSC_FREQ)][0])
    else:
        psd_ax.plot(freqs,pre_ap_fit, color = prestim_color, linewidth=3, linestyle='--')
        psd_ax.plot(freqs,pst_ap_fit, color = poststim_color, linewidth=3, linestyle='--')
        psd_ax.fill_between(
            freqs[alpha_mask], pows_post[alpha_mask], pst_ap_fit[alpha_mask], where=(pows_post[alpha_mask] != pst_ap_fit[alpha_mask]), 
            interpolate=True, color=poststim_color, alpha=0.25
            )
        psd_ax.fill_between(
            freqs[alpha_mask], pows_pre[alpha_mask], pre_ap_fit[alpha_mask], where=(pows_pre[alpha_mask] != pre_ap_fit[alpha_mask]), 
            interpolate=True, color=prestim_color, alpha=0.25
            )
        bar_ax.bar([u'Δ oscillation', u'Δ aperiodic'], height=[ flat_spec_delta, ap_delta], color='grey')
    # bar_ax.set_ylim(BAR_YLIM)
    bar_ax.set_ylabel('% Power Change')

    psd_ax.set_yscale('log')

if __name__ == "__main__":
    main()
