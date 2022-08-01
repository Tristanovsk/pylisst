import os
import numpy as np
from scipy.interpolate import interp1d

import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
from pylisst.process import process
from pylisst.driver import driver
from pylisst.calibration import calib
plt.ioff()
calfact = calib()


dir = '/DATA/projet/gernez/hablab/lisst_vsf'
file_ = 'V1111510.VSF'
#file_ = 'V1101425.VSF'
filez_ = 'Z1110820.VSF'

file = os.path.join(dir, file_)
filez = os.path.join(dir, filez_)
scat = driver(file)
scat.reader()
zsc = driver(filez)
zsc.reader()
p = process(scat, zsc, calfact)
p.full_process()

b = p
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 9), sharex=True, sharey=True)
fig.subplots_adjust(bottom=0.1, top=0.975, left=0.1, right=0.975,
                    hspace=0.1, wspace=0.1)
axs = axs.ravel()
b.rp.plot(hue='set', color='black', alpha=0.3, lw=1, add_legend=False, ax=axs[0])
b.rr.plot(hue='set', color='black', alpha=0.3, lw=1, add_legend=False, ax=axs[1])
b.pp.plot(hue='set', color='black', alpha=0.3, lw=1, add_legend=False, ax=axs[2])
b.pr.plot(hue='set', color='black', alpha=0.3, lw=1, add_legend=False, ax=axs[3])

axs[3].semilogy()
plt.savefig(os.path.join("fig", file_) + '.png', dpi=300)

fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(14, 5), sharex=True)
fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.975,
                    hspace=0.1, wspace=0.25)
axs = axs.ravel()
b.P11.plot(hue='set', color='black', alpha=0.3, lw=1, add_legend=False, ax=axs[0])
b.p12.plot(hue='set', color='black', alpha=0.3, lw=1, add_legend=False, ax=axs[1])
b.p22.plot(hue='set', color='black', alpha=0.3, lw=1, add_legend=False, ax=axs[2])

axs[0].semilogy()
for i in range(3):
    axs[i].minorticks_on()
plt.savefig(os.path.join("fig", file_) + '_mueller_mat.png', dpi=300)
