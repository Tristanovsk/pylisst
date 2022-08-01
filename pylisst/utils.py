import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

color_cycle = ['dimgrey', 'firebrick', 'darkorange', 'olivedrab',
               'dodgerblue', 'magenta']
plt.ioff()
plt.rcParams.update({'font.family': 'serif',
                     'font.size': 16, 'axes.labelsize': 20,
                     'mathtext.fontset': 'stix',
                     'axes.prop_cycle': plt.cycler('color', color_cycle)})
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


class plot:
    def __init__(self):
        pass

    def semilog(self, ax, size=2):
        ax.set_xlim((0.08, 5))
        divider = make_axes_locatable(ax)
        axlin = divider.append_axes("right", size=size, pad=0, sharey=ax)
        ax.spines['right'].set_visible(False)
        axlin.spines['left'].set_linestyle('--')
        # axlin.spines['left'].set_linewidth(1.8)
        axlin.spines['left'].set_color('grey')
        axlin.yaxis.set_ticks_position('right')
        axlin.yaxis.set_visible(False)
        axlin.xaxis.set_visible(False)
        axlin.set_xscale('linear')
        axlin.set_xlim((5, 170))
        ax.semilogy()
        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=4))
        ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=10, subs=np.arange(10) * 0.1))
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=10, subs=np.arange(10) * 0.1))
        return ax, axlin
