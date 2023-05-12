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

dir = '/DATA/projet/gernez/hablab/lisst_vsf/raw'
odir= '/DATA/projet/gernez/hablab/lisst_vsf/L2'

file_ = 'V1101629.VSF'
name = "PRY_C3"
filez_ = 'Z1110820.VSF'
info_file = opj(dir, 'LISST_VSF_Sample_Filenames_th.xlsx')
info = pd.read_excel(info_file)
info['c_mean']=np.nan
info['c_median']=np.nan
info['c_std']=np.nan

for irow, sample in info.iterrows():

    file_ = sample.filename

    if file_[0] == 'Z':
        continue
    name = sample.ID
    attrs = name.split('_')
    if name == 'MilliQ':
        attrs = ['MilliQ','','']
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


    c_mean,c_median,c_std = float(p.beam_c.mean()),float(p.beam_c.median()), float(p.beam_c.std())
    info.loc[irow,['c_mean','c_median','c_std']]=c_mean,c_median,c_std

    output = xr.Dataset({
                'name': attrs[0],
                'type': attrs[1],
                'number': attrs[2],
                'beam_c':p.beam_c,
                'P11':p.P11,
                'p12':p.p12,
                'p22':p.p22})

    output.to_netcdf(opj(odir, name) + '_scat_mat.nc')
