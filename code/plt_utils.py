"""
Plotting utility functions and settings.
"""

# plotting settings
FIGURE_WIDTH = 7 # Nature Reviews maximum figure size: 180mm (w) x 215 mm (h)
PANEL_FONTSIZE = 12


def remove_spines(ax):
    """
    Remove the top and left spines from a matplotlib axis.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
