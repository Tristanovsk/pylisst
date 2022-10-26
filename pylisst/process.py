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
    def __init__(self, scat, zsc, calfact, alpha=None):
        self.scat = scat
        self.zsc = zsc
        self.calfact = calfact
        self.alpha = alpha
        self.proc_date = dt.datetime.utcnow()
        self.version = "v1.0"
        self.cuvette_length = 15  # in cm
        self.eyeball_length = 10  # in cm
        self.water_refactive_index = 1.334
        self.eyeball_angle_min = 14
        self.eyeball_angle_max = 156

    def auxdata(self):
        # correct and save auxiliary data
        scat =self.scat
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
        self.sampl_t = self.scat.pow_trns / (self.zscat_r * self.scat.pow_lref)
        self.ringc = - np.log(self.sampl_t) / (1e-2 * self.cuvette_length)

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

    def process_large_angles(self):
        '''
        Correct ring and eyeball signals for laser ref and attenuation along path
        :return: 
        '''
        zsc = self.zsc
        scat = self.scat
        calfact = self.calfact
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
        self.scale_factor = scale_factor
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
        self.beam_c = - np.log(self.tau) / (1e-2 * self.cuvette_length)

        # attenuation correction along beam + from SV to eyeball
        beam_c1 = self.beam_c.isel(config=0)
        att_factors1 = np.exp(-beam_c1 * paths)
        # Scale signal to account for attenuation along path and laser power
        rp = scat.rp / att_factors1
        # Subtract background scattering
        rp = rp - self.zsc_rp
        # Correct for scattering volume lengthening with view angle
        rp = rp * np.sin(angles_rad)

        rr = scat.rr / att_factors1
        rr = rr - self.zsc_rr
        rr = rr * np.sin(angles_rad)

        # -----------------------------------------------------------------
        # Second rotation is with 1/2-wave plate to rotate polarization parallel
        # Signals are pp and pr (b and d). Processing is the same as above.
        # -----------------------------------------------------------------
        beam_c2 = self.beam_c.isel(config=1)
        att_factors2 = np.exp(-beam_c2 * paths)

        pp = scat.pp / att_factors2
        pp = pp - self.zsc_pp
        pp = pp * np.sin(angles_rad)

        pr = scat.pr / att_factors2
        pr = pr - self.zsc_pr
        pr = pr * np.sin(angles_rad)

        # Correct change in laser power
        rp[:, :40] = rp[:, :40] * calfact.laser_power_change_factor
        pp[:, :40] = pp[:, :40] * calfact.laser_power_change_factor
        rr[:, :40] = rr[:, :40] * calfact.laser_power_change_factor
        pr[:, :40] = pr[:, :40] * calfact.laser_power_change_factor

        # Interpolate over laser gain transition
        # TODO check if necessary or can be replaced with masking
        # rp(:,[40 41]) = interp1(angles([39 42]),rp(:,[39 42])',angles([40 41]))'
        # pp(:,[40 41]) = interp1(angles([39 42]),pp(:,[39 42])',angles([40 41]))'
        # rr(:,[40 41]) = interp1(angles([39 42]),rr(:,[39 42])',angles([40 41]))'
        # pr(:,[40 41]) = interp1(angles([39 42]),pr(:,[39 42])',angles([40 41]))'

        # geometric correction for slight misalignment between laser and eyeball viewing plane
        # TODO not sure is misalignment, maybe correction for scattering volume
        geom_corr = np.polyval(self.calfact.geometric_cal_coeff, self.angles)
        geom_corr = xr.DataArray(geom_corr, dims='angles',
                                 coords={'angles': self.angles})
        self.geom_corr = geom_corr
        self.rp = rp * geom_corr
        self.rr = rr * geom_corr
        self.pp = pp * geom_corr
        self.pr = pr * geom_corr

        # if did not provide alpha as input parameter, so estimate it using data
        if self.alpha == None:
            self.get_alpha()

        # save parameters
        self.rp = rp
        self.rr = rr * self.alpha
        self.pp = pp
        self.pr = pr * self.alpha

    def process_forward_angles(self):
        '''
        Process ring data to near-forward VSF
        :return:
        '''
        # ring radii in mm
        self.ring_radii = np.logspace(0, np.log10(200), 33) * 0.1
        # ring angles in water in radians
        # TODO understand the value "53"
        self.ring_angles = np.arcsin(np.sin(np.arctan(self.ring_radii / 53)) / self.water_refactive_index)

        # find solid angle; factor 6 takes care of rings covering only 1/6th circle
        cos_angles = np.cos(self.ring_angles)
        dOmega = cos_angles[:32] - cos_angles[1: 33]
        dOmega = dOmega * 2 * np.pi / 6
        dOmega = xr.DataArray(dOmega, dims=["number"],
                              coords=dict(number=range(self.calfact.number_rings)),
                              attrs=dict(description="LISST-VSF solid angles of rings",
                                         units="sr"))

        # Calculating scat and cscat according to "Processing LISST-100 and LISST-100X data
        scat1 = self.scat.rings1 / self.tau.isel(config=0)
        scat1 = scat1 - self.zsc_rings1 * self.scale_factor.isel(config=0)
        cscat1 = scat1 * self.calfact.dcal * self.calfact.dvig * self.calfact.ND

        # ring area correction, vignetting correction, ND filter transm. corr.
        scat2 = self.scat.rings1 / self.tau.isel(config=1)
        scat2 = scat2 - self.zsc_rings2 * self.scale_factor.isel(config=1)
        cscat2 = scat2 * self.calfact.dcal * self.calfact.dvig * self.calfact.ND

        light_on_rings1 = cscat1 * self.calfact.Watt_per_count_on_rings
        light_on_rings2 = cscat2 * self.calfact.Watt_per_count_on_rings

        # calculate incident laser power from LREF
        laser_incident_power1 = self.scat.LREF[:, 0] * self.calfact.Watt_per_count_laser_ref
        laser_incident_power2 = self.scat.LREF[:, 1] * self.calfact.Watt_per_count_laser_ref

        # calculate forward scattering; factor 6 due to arcs
        beam_bf = 6 * 0.5 * (np.sum(light_on_rings1 / laser_incident_power1) \
                             + np.sum(light_on_rings2 / laser_incident_power2))
        self.beam_bf = beam_bf / (1e-2 * self.cuvette_length)

        # calculate VSF for ring angles
        rho = 200 ** (1. / 32)
        self.ring_angles = self.ring_angles[:32] * np.sqrt(rho)
        self.ring_angles_deg = np.degrees(self.ring_angles)
        self.ring_cscat1 = cscat1
        self.ring_cscat2 = cscat2
        self.vsf1 = light_on_rings1 / (1e-2 * self.cuvette_length * dOmega * laser_incident_power1)
        self.vsf2 = light_on_rings2 / (1e-2 * self.cuvette_length * dOmega * laser_incident_power2)

        # remove (mask as np.nan) ring data that has very low signal
        mask = (scat1 > 25) | (scat2 > 25)
        self.vsf1 = self.vsf1.where(mask)
        self.vsf2 = self.vsf2.where(mask)
        self.ring_vsf = 0.5 * (self.vsf1 + self.vsf2)
        # reformat to get array with dimension with angle values
        self.ring_vsf = self.ring_vsf.rename({'number': 'angles'}).assign_coords({"angles": self.ring_angles_deg})

    def merge_angles(self):
        '''
        Scale vsf from overlapping angles
        (from Sequoia 15-16 deg, but here it is preferred not to extrapolate ring data)
        :return:
        '''

        ang_overlap = [13, 14]
        # TODO improve scaling, here median over two angles!!!
        scale_factor = (self.ring_vsf.interp(angles=ang_overlap) / self.p11.interp(angles=ang_overlap)).median(dim='angles')
        self.p11 = scale_factor * self.p11
        self.p11_scale_factor = scale_factor

        # truncate eyeball data to the usuable range
        valid_mask =slice(self.eyeball_angle_min,self.eyeball_angle_max)
        self.p11 = self.p11.sel(angles=valid_mask)
        self.p12 = self.p12.sel(angles=valid_mask)
        self.p22 = self.p22.sel(angles=valid_mask)
        self.P11 = self.ring_vsf.combine_first(self.p11)


    def compute_scattering_coef(self):
        # TODO
        return np.trapz(self.p11)

    def get_matrix_terms(self):
        '''
        Compute Mueller matrix terms: P11, P12, P22
        :return:
        '''
        rp, pp, rr, pr = self.rp, self.pp, self.rr, self.pr
        # rp, pp,rr,pr = self.rp, self.pp, self.rr_scaled,self.pr_scaled

        # P11
        p11 = 0.25 * (rp + pp + rr + pr)

        # P12
        p12 = 0.25 * ((pp - rp) + (pr - rr)) / p11

        # Extract p22
        phi = np.radians(self.angles)
        cos2phi = np.cos(2 * phi)
        e = rp * (1 + cos2phi)
        f = pp * (1 - cos2phi)
        g = rr * (1 - cos2phi)
        h = pr * (1 + cos2phi)
        # Two different estimates of P22
        # p22_1=(2*p11+(e+f))./(1+cos(4*repmat(p,ns,1)*pi/180))./p11
        # p22_2=(2*p11+(g+h))./(1+cos(4*repmat(p,ns,1)*pi/180))./p11
        p22_1 = ((2 * p11 - (e + f)) / (2 * cos2phi) ** 2) / p11
        p22_2 = ((2 * p11 - (g + h)) / (2 * cos2phi) ** 2) / p11
        # remove corrupted data
        mask = ~((self.angles >= 40) & (self.angles <= 50) | (self.angles >= 130) & (self.angles <= 140))
        p22_1 = p22_1.where(mask)
        p22_2 = p22_2.where(mask)
        self.p11 = p11
        self.p12 = p12
        self.p22 = (p22_1 + p22_2)

    def get_alpha(self):
        '''
        Alpha is the relative gain of the two photomultipliers.
        Alpha is not known a priori, it is determined from data.

        :return:
        '''
        ang_ref1 = 45
        ang_ref2 = 135

        alpha_ac45 = self.rp.sel(angles=ang_ref1) / self.rr.sel(angles=ang_ref1)
        alpha_ac135 = self.rp.sel(angles=ang_ref2) / self.rr.sel(angles=ang_ref2)
        alpha_bd45 = self.pp.sel(angles=ang_ref1) / self.pr.sel(angles=ang_ref1)
        alpha_bd135 = self.pp.sel(angles=ang_ref2) / self.pr.sel(angles=ang_ref2)
        self.alpha = np.nanmedian([alpha_ac45, alpha_ac135, alpha_bd45, alpha_bd135])

    def full_process(self):
        self.auxdata()
        self.get_attenuation()
        self.get_angles()
        self.process_large_angles()
        self.process_forward_angles()
        self.get_matrix_terms()
        self.merge_angles()

