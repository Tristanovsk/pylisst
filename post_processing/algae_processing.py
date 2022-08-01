import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rc = {"font.family": "serif",
      "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18})

from pylisst.process import process
from pylisst.driver import driver
from pylisst.calibration import calib
import pylisst.utils as utils
# personal package
from mie_perso import psd, mie_multiprocess

opj = os.path.join

calfact = calib()

mie = False
dir = '/DATA/projet/gernez/hablab/lisst_vsf'

file_ = 'V1101629.VSF'
name = "PRY_C3"
filez_ = 'Z1110820.VSF'
info_file = opj(dir, 'LISST_VSF_Sample_Filenames_th.xlsx')
info = pd.read_excel(info_file)
info['c_mean']=np.nan
info['c_median']=np.nan
info['c_std']=np.nan

for irow, sample in info.iterrows():
    mie = False
    file_ = sample.filename

    if file_[0] == 'Z':
        continue
    name = sample.ID
    if 'BEADS' in name:
        mie = True
        rn_med = sample.median_radius
    # else:
    #     continue
    print(sample.Date, name)
    # ---------------------------------
    # open and process LISST-VSF data
    # ---------------------------------
    file = os.path.join(dir, file_)
    filez = os.path.join(dir, filez_)
    scat = driver(file)
    scat.reader()
    zsc = driver(filez)
    zsc.reader()
    p = process(scat, zsc, calfact)
    p.full_process()

    if mie:
        # ---------------------------------
        # get Mie simulations
        # ---------------------------------
        pmie = mie_multiprocess.processor()
        size_param = psd.size_param
        psd_ = psd.psd()

        # -------------------------------------
        # Generate mueller matrices for a series
        # of size parameters x = np.pi * diameter / wavelength (unitless)
        # -------------------------------------
        wl = 515
        nMedium = 1.3199 + 6878 / wl ** 2 - 1.132e9 / wl ** 4 + 1.11e14 / wl ** 6
        wl_medium = wl / nMedium
        npolystyrene = 1.60
        m = npolystyrene / nMedium - 0.000j

        theta = np.linspace(0, np.pi, 3600)
        x = np.logspace(np.log10(1), 2, 1001)
        ofile = 'data/mueller_mie_' + format(m, '1.3f') + '_t3600.nc'

        if os.path.exists(ofile):
            mueller = xr.open_dataset(ofile)
        else:
            mueller = pmie.ScatMat_mp(m, x, theta)
            mueller.to_netcdf(ofile)

        # -------------------------------------
        # convert mueller matrices for a series
        # of diameters for a given couple of
        # wavelength (in vacuum) and refractive index (in medium)
        # for a medium of refractive index nMedium
        # -------------------------------------
        theta = mueller.theta
        ang = theta * 180. / np.pi
        Ntheta = len(mueller.theta)

        m = complex(mueller.nr, mueller.ni)

        m_vacuum = m * nMedium
        wl_medium = wl / nMedium

        dpnm = mueller.x * wl_medium / np.pi
        dp = dpnm / 1000

        CV = 0.01
        sig = rn_med * CV
        rv_med = psd_.rnmed2rvmed(rn_med, sig)

        ndp = psd_.lognorm(dp / 2, rn_med=rn_med, sigma=sig)
        # ndp = modif_power_law(dp / 2, slope=-slope, rmin=rmin, rmax=rmax)
        # convert to xarray
        ndp = dp.copy(data=ndp)

        # ndp = psd #[:-1]*np.diff(dp/2)
        S11, S12, S33, S34 = np.zeros(Ntheta), np.zeros(Ntheta), np.zeros(Ntheta), np.zeros(Ntheta)

        # aSDn = np.pi*((dp/2)**2)*ndp
        aSDn = ndp
        S11 = np.trapz(mueller.S11 * aSDn, dp, axis=0)
        S12 = np.trapz(mueller.S12 * aSDn, dp, axis=0) / S11
        S33 = np.trapz(mueller.S33 * aSDn, dp, axis=0) / S11
        S34 = np.trapz(mueller.S34 * aSDn, dp, axis=0) / S11
        norm = np.trapz(S11 * np.sin(theta), theta) / 2

    b = p
    c_mean,c_median,c_std = float(b.beam_c.mean()),float(b.beam_c.median()), float(b.beam_c.std())
    info.loc[irow,['c_mean','c_median','c_std']]=c_mean,c_median,c_std
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(16, 6), )
    fig.subplots_adjust(bottom=0.15, top=0.8, left=0.1, right=0.975,
                        hspace=0.1, wspace=0.25)
    axs = axs.ravel()
    ax = axs[0]
    ax.loglog()
    ax, axlin = utils.plot().semilog(ax, size=3.1)
    for ax_ in (ax, axlin):
        b.P11.plot(hue='set', color='black', alpha=0.3, lw=1, add_legend=False, ax=ax_)

    b.p12.plot(hue='set', color='black', alpha=0.3, lw=1, add_legend=False, ax=axs[1])
    b.p22.plot(hue='set', color='black', alpha=0.3, lw=1, add_legend=False, ax=axs[2])

    if mie:
        ax.plot(ang, S11 / norm, c='red', label='Mie')
        axlin.plot(ang, S11 / norm, c='red', label='Mie')
        axs[1].plot(ang, S12, c='red')
        axs[2].hlines(1, 0, 180, colors='red')
        handles, labels = axlin.get_legend_handles_labels()
        patch = Line2D([0], [0], color='black', label='LISST-VSF data')
        handles.append(patch)
        axs[2].legend(handles=handles, loc='lower left')

    axs[1].set_title('$P_{12}$')
    axs[2].set_title('$P_{22}$')
    for i in range(1, 3):
        axs[i].set_xlabel('$Scattering\ angle\ (deg)$')
    axlin.text(0.95, 0.95,
               r'$c=${:6.3f}$\pm${:5.3f}'.format(c_median, c_std) + ' $m^{-1}$',
               size=18, transform=axlin.transAxes, ha="right", va="top", )
    axlin.xaxis.set_visible(True)
    ax.set_xlabel('')
    axlin.set_xlabel('$Scattering\ angle\ (deg)$')
    axlin.set_title('$P_{11}$')
    plt.suptitle(name)
    plt.show()
    plt.savefig(os.path.join("fig", name) + '_mueller_mat_mie_check.png', dpi=300)
    plt.close()


info.to_csv('lisst_vsf_attenuation.csv',index=False)