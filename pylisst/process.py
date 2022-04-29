import os
import numpy as np
from scipy.interpolate import interp1d

import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt

from pylisst.driver import driver
from pylisst.calibration import calib

calfact = calib()


class process:
    def __init__(self, scat, zsc, calfact):
        self.scat = scat
        self.zsc = zsc
        self.calfact = calfact
        self.proc_date = dt.datetime.utcnow()
        self.version = "v1.0"
        self.cuvette_length = 15  # in cm
        self.eyeball_length = 10  # in cm

    def auxdata(self):
        # correct and save auxiliary data
        self.timestamp = scat.time
        self.tempC = scat.tempC * calfact.temp_slope + calfact.temp_offset
        self.depth = scat.depth * calfact.depth_slope + calfact.depth_offset
        self.batt_volts = scat.batt_volts * calfact.bat_slope + calfact.bat_offset
        self.pmt_gain = scat.pmt_gain

    def get_attenuation(self):
        # EDIT added code to extract 'classic' attenuation from bins 1 - 40(791 - 830)
        zscr = self.zsc.pow_trns / self.zsc.pow_lref
        # TODO check data filtering/smoothing/flagging ... method
        self.zscr_clean = zscr[abs(zscr - np.median(zscr)) < np.std(zscr)]
        self.zscat_r = np.median(self.zscr_clean)
        self.sampl_t = self.scat.pow_trns / (self.zscat_r * scat.pow_lref)
        self.ringc = -100 / 15 * np.log(self.sampl_t)

        ns, nang = scat.rp.shape

    def get_angles(self):
        # Convert encoder index in datastream from ADC board to actual angle, offset is contained in Cal_Factors file
        # angles in degrees
        angles = self.scat.angles_idx + self.calfact.angle_offset
        # TODO try to understand why testinf=g saturation for the range 14 to 156 ???
        self.qc_saturated = (self.scat.qc_saturated[:, (angles > 14) & (angles < 156)]).any(axis=1)
        self.angles = angles
        self.update_coords(angles)

    def update_coords(self, angles):
        self.scat.rp = self.scat.rp.assign_coords(angles=angles)
        self.scat.rr = self.scat.rr.assign_coords(angles=angles)
        self.scat.pp = self.scat.pp.assign_coords(angles=angles)
        self.scat.pr = self.scat.pr.assign_coords(angles=angles)
        self.zsc.rp = self.zsc.rp.assign_coords(angles=angles)
        self.zsc.rr = self.zsc.rr.assign_coords(angles=angles)
        self.zsc.pp = self.zsc.pp.assign_coords(angles=angles)
        self.zsc.pr = self.zsc.pr.assign_coords(angles=angles)

    def attenuation_correction(self):
        '''
        Correct ring and eyeball signals for laser ref and attenuation along path
        :return: 
        '''
        zsc = self.zsc
        scat = self.scat
        HWPlate_transmission = self.calfact.HWPlate_transmission
        angles_rad = np.radians(self.angles)

        # Correct raw measurements for the reduction in laser power 
        # caused by the 1/2 wave plate
        zsc.LREF[:, 1] = zsc.LREF[:, 1] * HWPlate_transmission
        scat.LREF[:, 1] = scat.LREF[:, 1] * HWPlate_transmission
        scat.rp = scat.rp * HWPlate_transmission
        scat.rr = scat.rr * HWPlate_transmission
        zsc.rp = zsc.rp * HWPlate_transmission
        zsc.rr = zsc.rr * HWPlate_transmission

        # Find number of PMT gain values in background file (usually 10)
        zsc_pmt_values = np.unique(zsc.pmt_gain)
        num_zsc_pmts = len(zsc_pmt_values)

        self.zsc_rp = zsc.rp.groupby('pmt').median()
        self.zsc_rr = zsc.rr.groupby('pmt').median()
        self.zsc_pp = zsc.pp.groupby('pmt').median()
        self.zsc_pr = zsc.pr.groupby('pmt').median()
        self.zsc_LP = zsc.LP.groupby('pmt').median()
        self.zsc_LREF = zsc.LREF.groupby('pmt').median()
        self.zsc_rings1 = zsc.rings1.groupby('pmt').median()
        self.zsc_rings2 = zsc.rings2.groupby('pmt').median()
        self.zsc_pmt_gain = np.unique(zsc.pmt_gain)

        # reproject on actual number of angle, i.e., scat.pmt_gain
        self.zsc_rp = self.zsc_rp.interp(pmt=scat.pmt_gain, method='nearest')
        self.zsc_rr = self.zsc_rr.interp(pmt=scat.pmt_gain, method='nearest')
        self.zsc_pp = self.zsc_pp.interp(pmt=scat.pmt_gain, method='nearest')
        self.zsc_pr = self.zsc_pr.interp(pmt=scat.pmt_gain, method='nearest')
        self.zsc_LP = self.zsc_LP.interp(pmt=scat.pmt_gain, method='nearest')
        self.zsc_LREF = self.zsc_LREF.interp(pmt=scat.pmt_gain, method='nearest')
        self.zsc_rings1 = self.zsc_rings1.interp(pmt=scat.pmt_gain, method='nearest')
        self.zsc_rings2 = self.zsc_rings2.interp(pmt=scat.pmt_gain, method='nearest')
        self.zsc_pmt_gain = zsc.pmt_gain

        # -----------------------------------------------------------------
        # distance along the beam to sample volume, then from the sample volume to eyeball [cm]
        # used for attenuation correction later
        #                 Eyeball (.)
        #                         /
        #                        /
        #  Receive Window |     ------------------| Transmit Window
        #                       ^
        #                 Sample Volume
        # -----------------------------------------------------------------
        paths = self.eyeball_length - 2 * np.arctan(angles_rad) \
                + 2. / np.sin(angles_rad)
        # convert in meter
        paths = xr.DataArray(paths * 1e-2, dims='angles',
                             coords={'angles': self.angles})
        scale_factor = self.scat.LREF / self.zsc_LREF
        # replicate scale factor for every angle
        # scale_factor =scale_factor.expand_dims({'angles': self.angles})
        self.zsc_rp = self.zsc_rp * scale_factor.isel(config=0)
        self.zsc_rr = self.zsc_rr * scale_factor.isel(config=0)
        self.zsc_pp = self.zsc_pp * scale_factor.isel(config=1)
        self.zsc_pr = self.zsc_pr * scale_factor.isel(config=1)

        # clean water ratio of transmitted laser power to reference; used to correct for laser drift
        drift_corr = self.zsc_LP / self.zsc_LREF

        # -----------------------------------------------------------------
        # First rotation is laser polarized perpendicular, signals are then
        # rp and rr (a and c)
        # -----------------------------------------------------------------
        # laser reference drift compensated here.
        self.tau = scat.LP / (drift_corr * scat.LREF)
        self.beam_c = -100. / self.cuvette_length * np.log(self.tau)

        # attenuation correction along beam + from SV to eyeball
        beam_c1 = self.beam_c.isel(config=0)
        att_factors1 = np.exp(-beam_c1 * paths)
        # Scale signal to account for attenuation along path and laser power
        rp = scat.rp / att_factors1
        # Subtract background scattering
        rp = rp - self.zsc_rp
        # Correct for scattering volume lengthening with view angle
        rp = rp * np.sin(angles_rad)
        self.rp = rp

        rr = scat.rr / att_factors1
        rr = rr - self.zsc_rr
        rr = rr * np.sin(angles_rad)
        self.rr = rr

        # -----------------------------------------------------------------
        # Second rotation is with 1/2-wave plate to rotate polarization parallel
        # Signals are pp and pr (b and d). Processing is the same as above.
        # -----------------------------------------------------------------
        beam_c2 = self.beam_c.isel(config=1)
        att_factors2 = np.exp(-beam_c2 * paths)

        pp = scat.pp / att_factors2
        pp = pp - self.zsc_pp
        pp = pp * np.sin(angles_rad)
        self.pp = pp
        pr = scat.pr / att_factors2
        pr = pr - self.zsc_pr
        pr = pr * np.sin(angles_rad)
        self.pr = pr



dir = '/DATA/projet/gernez/hablab/lisst_vsf'
file_ = 'V1111510.VSF'
# file_ = 'V1101425.VSF'
filez_ = 'Z1110820.VSF'
file = os.path.join(dir, file_)
filez = os.path.join(dir, filez_)
scat = driver(file)
scat.reader()
zsc = driver(filez)
zsc.reader()
p = process(scat, zsc, calfact)
p.auxdata()
p.get_attenuation()
p.get_angles()
p.attenuation_correction()

b = p
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 9), sharex=True, sharey=True)
fig.subplots_adjust(bottom=0.1, top=0.975, left=0.1, right=0.975,
                    hspace=0.1, wspace=0.1)
axs = axs.ravel()
b.rp.plot(hue='set', ax=axs[0])
b.rr.plot(hue='set', ax=axs[1])
b.pp.plot(hue='set', ax=axs[2])
b.pr.plot(hue='set', ax=axs[3])

axs[3].semilogy()
plt.savefig(os.path.join("fig", file_) + '.png', dpi=300)
