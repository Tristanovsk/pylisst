"""
Processing of LISST-VSF data into volume scattering function and Mueller matrix terms.
"""

import os
import numpy as np
from scipy.interpolate import interp1d

import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt

from pylisst.driver import driver
from pylisst.calibration import calib
from pylisst import __version__

calfact = calib()


class process:
    """
    Processing chain of LISST-VSF measurements.

    Converts raw sample and clean-water background measurements into the
    beam attenuation coefficient, the volume scattering function (VSF)
    over near-forward (ring detectors) and large (eyeball) angles, and the
    Mueller matrix terms P11, P12 and P22.

    Parameters
    ----------
    scat : pylisst.driver.driver
        Sample measurement, already read with :meth:`~pylisst.driver.driver.reader`.
    zsc : pylisst.driver.driver
        Clean-water background measurement, already read with
        :meth:`~pylisst.driver.driver.reader`.
    calfact : pylisst.calibration.calib
        Instrument calibration factors.
    alpha : float, optional
        Relative gain of the two eyeball photomultipliers. If None (default),
        it is estimated from data with :meth:`get_alpha`.

    Attributes
    ----------
    cuvette_length : float
        Optical path length of the sample chamber (m).
    eyeball_length : float
        Distance from the transmit window to the eyeball sample volume (m).
    water_refactive_index : float
        Refractive index of water used to convert ring angles in air to water.
    eyeball_angle_min, eyeball_angle_max : float
        Valid angular range (degrees) of the eyeball data.
    ang_overlap : numpy.ndarray
        Angles (degrees) used to scale eyeball data onto ring data.
    drop_eyeball_angle : float
        Eyeball angle (degrees) dropped and re-interpolated due to eyeball design.

    Notes
    -----
    Main outputs after :meth:`full_process`:

    - ``beam_c``: beam attenuation coefficient (m-1), dimensions ``(set, config)``;
    - ``ring_vsf``: VSF from the ring detectors (m-1 sr-1);
    - ``p11``, ``p12``, ``p22``: eyeball Mueller matrix terms (P12 and P22
      normalized by P11);
    - ``P11``: VSF merged from rings and eyeball over the full angular range;
    - ``alpha``: relative gain of the two photomultipliers.

    Examples
    --------
    >>> from pylisst import driver, calib, process
    >>> scat = driver('V1111510.VSF'); scat.reader()
    >>> zsc = driver('Z1110820.VSF'); zsc.reader()
    >>> p = process(scat, zsc, calib())
    >>> p.full_process()
    >>> p.P11.plot(hue='set')
    """

    def __init__(self, scat, zsc, calfact, alpha=None):
        self.scat = scat
        self.zsc = zsc
        self.calfact = calfact
        self.alpha = alpha
        self.proc_date = dt.datetime.utcnow()
        self.version = __version__
        self.cuvette_length = 0.15  # in m
        self.eyeball_length = 0.10  # in m
        self.water_refactive_index = 1.334
        self.eyeball_angle_min = 14
        self.eyeball_angle_max = 154 # 156
        self.ang_overlap = np.linspace(14, 15, 10)
        # angle to drop and reinterpolate due to eyeball design
        self.drop_eyeball_angle= 48

    def auxdata(self):
        """
        Convert auxiliary data into physical units.

        Sets attributes ``timestamp``, ``tempC`` (degrees Celsius),
        ``depth`` (m), ``batt_volts`` (V) and ``pmt_gain``.
        """
        # correct and save auxiliary data
        scat = self.scat
        self.timestamp = scat.time
        self.tempC = scat.tempC * calfact.temp_slope + calfact.temp_offset
        self.depth = scat.depth * calfact.depth_slope + calfact.depth_offset
        self.batt_volts = scat.batt_volts * calfact.bat_slope + calfact.bat_offset
        self.pmt_gain = scat.pmt_gain

    def get_attenuation(self):
        """
        Compute the beam attenuation coefficient.

        The ratio of transmitted laser power to laser reference of the sample
        is normalized by the same ratio for clean water (median per PMT gain),
        which compensates for laser drift. Sets attributes ``tau``
        (transmittance) and ``beam_c`` (attenuation coefficient, m-1) with
        dimensions ``(set, config)``, as well as ``ringc``, the "classic"
        attenuation from the ring transmission detector.
        """

        # EDIT added code to extract 'classic' attenuation from bins 1 - 40(791 - 830)
        zscr = self.zsc.pow_trns / self.zsc.pow_lref

        # TODO check data filtering/smoothing/flagging ... method
        self.zscr_clean = zscr[abs(zscr - np.median(zscr)) < np.std(zscr)]
        self.zscat_r = np.median(self.zscr_clean)
        self.sampl_t = self.scat.pow_trns / (self.zscat_r * self.scat.pow_lref)
        self.ringc = - np.log(self.sampl_t) / self.cuvette_length

        # -----------------------------------------------------------------
        # set instrument parameter and get attenuation coef.
        # -----------------------------------------------------------------
        zsc = self.zsc
        scat = self.scat
        HWPlate_transmission = self.calfact.HWPlate_transmission

        # Correct raw measurements for the reduction in laser power
        # caused by the 1/2 wave plate
        zsc.LREF[:, 1] = zsc.LREF[:, 1] * HWPlate_transmission

        self.zsc_LP = zsc.LP.groupby('pmt').median()
        self.zsc_LREF = zsc.LREF.groupby('pmt').median()

        # reproject on actual number of angle, i.e., scat.pmt_gain
        self.zsc_LP = self.zsc_LP.interp(pmt=scat.pmt_gain, method='nearest')
        self.zsc_LREF = self.zsc_LREF.interp(pmt=scat.pmt_gain, method='nearest')

        # clean water ratio of transmitted laser power to reference; used to correct for laser drift
        drift_corr = self.zsc_LP / self.zsc_LREF

        # laser reference drift compensated here and attenuation coef. (beam_c)
        self.tau = scat.LP / (drift_corr * scat.LREF)
        self.beam_c = - np.log(self.tau) / self.cuvette_length


    def get_angles(self):
        """
        Convert eyeball encoder indices into scattering angles (degrees).

        Applies the calibration angle offset, updates the ``angles``
        coordinate of sample and background signals and sets ``qc_saturated``,
        True for sets with saturated signal between 14 and 156 degrees.
        """
        # Convert encoder index in datastream from ADC board to actual angle, offset is contained in Cal_Factors file
        # angles in degrees
        angles = self.scat.angles_idx + self.calfact.angle_offset-1
        # TODO try to understand why testing saturation for the range 14 to 156 ???
        self.qc_saturated = (self.scat.qc_saturated[:, (angles > 14) & (angles < 156)]).any(axis=1)
        self.angles = angles
        self.update_coords(angles)

    def get_scattering_volume(self):
        """
        Compute the eyeball geometric correction factor.

        Evaluates the calibration polynomial ``geometric_cal_coeff`` at the
        eyeball angles and stores it in ``geom_corr``.
        """
        # geometric correction for slight misalignment between laser and eyeball viewing plane
        # TODO not sure is misalignment, maybe correction for scattering volume
        geom_corr = np.polyval(self.calfact.geometric_cal_coeff, self.angles)
        geom_corr = xr.DataArray(geom_corr, dims='angles',
                                 coords={'angles': self.angles})
        self.geom_corr = geom_corr

    def update_coords(self, angles):
        """
        Assign new angle coordinates to sample and background eyeball signals.

        Parameters
        ----------
        angles : array_like
            Scattering angles (degrees).
        """
        self.scat.rp = self.scat.rp.assign_coords(angles=angles)
        self.scat.rr = self.scat.rr.assign_coords(angles=angles)
        self.scat.pp = self.scat.pp.assign_coords(angles=angles)
        self.scat.pr = self.scat.pr.assign_coords(angles=angles)
        self.zsc.rp = self.zsc.rp.assign_coords(angles=angles)
        self.zsc.rr = self.zsc.rr.assign_coords(angles=angles)
        self.zsc.pp = self.zsc.pp.assign_coords(angles=angles)
        self.zsc.pr = self.zsc.pr.assign_coords(angles=angles)

    def process_large_angles(self):
        """
        Correct eyeball signals for background, laser power and attenuation.

        Steps:

        1. correct for half-wave plate transmission;
        2. subtract the clean-water background (median per PMT gain,
           scaled by the laser reference ratio);
        3. compute ``tau`` and ``beam_c`` and correct for attenuation along the
           beam and from the sample volume to the eyeball;
        4. correct for scattering volume lengthening (``sin(angle)``);
        5. correct for the laser power change over the first 40 angles;
        6. apply the geometric correction ``geom_corr``;
        7. scale ``r`` detector signals by the PMT relative gain ``alpha``
           (estimated with :meth:`get_alpha` if not provided).

        Sets attributes ``rp``, ``rr``, ``pp`` and ``pr``.
        """
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
        paths = self.eyeball_length - 0.02 * np.arctan(angles_rad) \
                + 0.02 / np.sin(angles_rad)
        # convert in meter
        paths = xr.DataArray(paths, dims='angles',
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
        self.beam_c = - np.log(self.tau) / self.cuvette_length

        # attenuation correction along beam + from SV to eyeball
        beam_c1 = self.beam_c.isel(config=0)
        att_factors1 = np.exp(-beam_c1 * paths)

        # Subtract background scattering
        rp = scat.rp - self.zsc_rp
        # Scale signal to account for attenuation along path and laser power
        rp = rp / att_factors1
        # Correct for scattering volume lengthening with view angle
        rp = rp * np.sin(angles_rad)

        rr = scat.rr - self.zsc_rr
        rr = rr / att_factors1
        rr = rr * np.sin(angles_rad)

        # -----------------------------------------------------------------
        # Second rotation is with 1/2-wave plate to rotate polarization parallel
        # Signals are pp and pr (b and d). Processing is the same as above.
        # -----------------------------------------------------------------
        beam_c2 = self.beam_c.isel(config=1)
        att_factors2 = np.exp(-beam_c2 * paths)

        pp = scat.pp - self.zsc_pp
        pp = pp / att_factors2
        pp = pp * np.sin(angles_rad)

        pr = scat.pr - self.zsc_pr
        pr = pr / att_factors2
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
        # geom_corr = np.polyval(self.calfact.geometric_cal_coeff, self.angles)
        # geom_corr = xr.DataArray(geom_corr, dims='angles',
        #                          coords={'angles': self.angles})
        geom_corr = self.geom_corr

        self.rp = rp * geom_corr
        self.rr = rr * geom_corr
        self.pp = pp * geom_corr
        self.pr = pr * geom_corr

        # self.rp = rp
        # self.rr = rr
        # self.pp = pp
        # self.pr = pr

        # if did not provide alpha as input parameter, so estimate it using data
        if self.alpha == None:
            self.get_alpha()

        # save parameters
        self.rp = self.rp
        self.rr = self.rr * self.alpha
        self.pp = self.pp
        self.pr = self.pr * self.alpha

    def process_large_angles_vbeta(self):
        """
        Experimental version of :meth:`process_large_angles`.

        Identical to :meth:`process_large_angles` except that corrections for
        attenuation along path and for scattering volume lengthening are
        not applied.
        """
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
        paths = self.eyeball_length - 0.02 * np.arctan(angles_rad) \
                + 0.02 / np.sin(angles_rad)
        # convert in meter
        paths = xr.DataArray(paths, dims='angles',
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
        self.beam_c = - np.log(self.tau) / self.cuvette_length

        # attenuation correction along beam + from SV to eyeball
        beam_c1 = self.beam_c.isel(config=0)
        att_factors1 = np.exp(-beam_c1 * paths)

        # Subtract background scattering
        rp = scat.rp - self.zsc_rp
        # Scale signal to account for attenuation along path and laser power
        # rp = rp / att_factors1
        # Correct for scattering volume lengthening with view angle
        # rp = rp * np.sin(angles_rad)

        rr = scat.rr - self.zsc_rr
        # rr = rr / att_factors1
        # rr = rr * np.sin(angles_rad)

        # -----------------------------------------------------------------
        # Second rotation is with 1/2-wave plate to rotate polarization parallel
        # Signals are pp and pr (b and d). Processing is the same as above.
        # -----------------------------------------------------------------
        beam_c2 = self.beam_c.isel(config=1)
        att_factors2 = np.exp(-beam_c2 * paths)

        pp = scat.pp - self.zsc_pp
        # pp = pp / att_factors2
        # pp = pp * np.sin(angles_rad)

        pr = scat.pr - self.zsc_pr
        # pr = pr / att_factors2
        # pr = pr * np.sin(angles_rad)

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
        # geom_corr = np.polyval(self.calfact.geometric_cal_coeff, self.angles)
        # geom_corr = xr.DataArray(geom_corr, dims='angles',
        #                          coords={'angles': self.angles})
        geom_corr = self.geom_corr

        self.rp = rp * geom_corr
        self.rr = rr * geom_corr
        self.pp = pp * geom_corr
        self.pr = pr * geom_corr

        # self.rp = rp
        # self.rr = rr
        # self.pp = pp
        # self.pr = pr

        # if did not provide alpha as input parameter, so estimate it using data
        if self.alpha == None:
            self.get_alpha()

        # save parameters
        self.rp = self.rp
        self.rr = self.rr * self.alpha
        self.pp = self.pp
        self.pr = self.pr * self.alpha

    def process_forward_angles(self):
        """
        Process ring detector data into the near-forward VSF.

        Ring counts are corrected for attenuation, background, ring area,
        vignetting and neutral density filter, then converted to VSF using
        the ring solid angles in water and the incident laser power.
        Low-signal (< 25 counts) and negative values are masked.

        Sets attributes ``ring_angles_deg`` (degrees), ``vsf1``/``vsf2``
        (VSF for each laser polarization), ``ring_vsf`` (their average,
        m-1 sr-1, dimension ``angles``) and ``beam_bf`` (forward scattering
        coefficient, m-1).
        """
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
        self.beam_bf = beam_bf / self.cuvette_length

        # calculate VSF for ring angles
        rho = 200 ** (1. / 32)
        self.ring_angles = self.ring_angles[:32] * np.sqrt(rho)
        self.ring_angles_deg = np.degrees(self.ring_angles)
        self.ring_cscat1 = cscat1
        self.ring_cscat2 = cscat2
        self.vsf1 = light_on_rings1 / (self.cuvette_length * dOmega * laser_incident_power1)
        self.vsf2 = light_on_rings2 / (self.cuvette_length * dOmega * laser_incident_power2)
        trans = float(self.tau.median()) / (1 - np.cos(np.radians(self.ring_angles_deg)))
        # remove (mask as np.nan) ring data that has very low signal
        mask = (scat1 > 25) | (scat2 > 25)
        self.vsf1 = self.vsf1.where(mask)
        self.vsf2 = self.vsf2.where(mask)
        self.ring_vsf = 0.5 * (self.vsf1 + self.vsf2)
        # reformat to get array with dimension with angle values
        self.ring_vsf = self.ring_vsf.rename({'number': 'angles'}).assign_coords({"angles": self.ring_angles_deg})
        # mask negative or null values
        self.ring_vsf = self.ring_vsf.where(self.ring_vsf >= 0)

    def merge_angles(self,
                     custom_factor=1):
        """
        Merge ring and eyeball data into a VSF over the full angular range.

        Eyeball P11 is scaled to the ring VSF using the median ratio over the
        overlapping angles ``ang_overlap`` (Sequoia uses 15-16 deg, but here
        ring data are not extrapolated). Eyeball data are truncated to
        [``eyeball_angle_min``, ``eyeball_angle_max``], and the 48 deg angle is
        dropped and re-interpolated.

        Parameters
        ----------
        custom_factor : float, optional
            Additional multiplicative factor applied to the scaling factor.
            Default is 1.

        Notes
        -----
        Sets attributes ``P11`` (merged VSF) and ``p11_scale_factor`` and
        truncates ``p11``, ``p12`` and ``p22``.
        """

        # ang_overlap = [13, 14]  # default Sequoia
        ang_overlap = self.ang_overlap
        scale_factor = (self.ring_vsf.interp(angles=ang_overlap) /
                        self.p11.interp(angles=ang_overlap)).median(dim='angles')  * custom_factor

        self.p11 = scale_factor * self.p11
        self.p11_scale_factor = scale_factor

        # truncate eyeball data to the usuable range
        valid_mask = slice(self.eyeball_angle_min, self.eyeball_angle_max)
        self.p11 = self.p11.sel(angles=valid_mask)
        self.p12 = self.p12.sel(angles=valid_mask)
        self.p22 = self.p22.sel(angles=valid_mask)
        self.P11 = self.ring_vsf.combine_first(self.p11)
        full_angles = self.P11.angles
        self.P11 = self.P11.where(full_angles != 48, drop=True).interp(angles=full_angles)

    def compute_scattering_coef(self):
        """
        Compute the scattering coefficient.

        .. warning:: Not implemented yet (placeholder).
        """
        # TODO
        return np.trapz(self.p11)

    def get_matrix_terms(self):
        """
        Compute Mueller matrix terms P11, P12 and P22 from eyeball signals.

        P11 is the average of the four signals, P12 and P22 are normalized by
        P11. P22 is the mean of two estimates obtained from each laser
        polarization; values within 40-50 and 130-140 degrees are masked.

        Sets attributes ``p11``, ``p12`` and ``p22``.
        """
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
        """
        Estimate ``alpha``, the relative gain of the two photomultipliers.

        Alpha is not known a priori and is determined from data as the median
        ratio of ``p`` to ``r`` detector signals at 45 and 135 degrees, for
        both laser polarizations. Sets attribute ``alpha``.
        """
        ang_ref1 = 45
        ang_ref2 = 135

        alpha_ac45 = self.rp.sel(angles=ang_ref1) / self.rr.sel(angles=ang_ref1)
        alpha_ac135 = self.rp.sel(angles=ang_ref2) / self.rr.sel(angles=ang_ref2)
        alpha_bd45 = self.pp.sel(angles=ang_ref1) / self.pr.sel(angles=ang_ref1)
        alpha_bd135 = self.pp.sel(angles=ang_ref2) / self.pr.sel(angles=ang_ref2)
        self.alpha = np.nanmedian([alpha_ac45, alpha_ac135, alpha_bd45, alpha_bd135])

    def get_TdV(self,
                c_coef,
                Lin=0.1,
                Lout=0.05,
                R=0.02,
                wd=0.005,
                beta=0):
        """
        Compute transmittance and scattering volume for each eyeball angle.

        Parameters
        ----------
        c_coef : float or xarray.DataArray
            Beam attenuation coefficient (m-1).
        Lin : float, optional
            Distance from the transmit window to the eyeball axis (m). Default 0.1.
        Lout : float, optional
            Distance from the eyeball axis to the receive window (m). Default 0.05.
        R : float, optional
            Distance from the eyeball center to the sample volume (m). Default 0.02.
        wd : float, optional
            Width of the eyeball field of view (m). Default 0.005.
        beta : float, optional
            Tilt angle of the eyeball (degrees). Default 0.

        Returns
        -------
        numpy.ndarray or xarray.DataArray
            Product of ``1 - exp(-c_coef * path)`` and the length of the
            sample volume, for each eyeball angle.
        """

        D = Lin + Lout

        ang_rad = np.radians(self.angles)
        beta = np.radians(beta)
        alpha = (np.pi / 2 - ang_rad + beta)
        R = R * np.cos(beta)

        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        tan_alpha = np.tan(alpha)

        path_laser = Lin - R * tan_alpha
        path_dectect = R / cos_alpha

        length = wd / cos_alpha
        length

        delta_large_angle = D - (path_laser + length / 2)
        delta_large_angle[delta_large_angle > 0] = 0

        delta_small_angle = path_laser - length / 2
        delta_small_angle[delta_small_angle > 0] = 0

        length = length  # +delta_large_angle +delta_small_angle
        # length[length<0]=0.00
        dV = length

        # path_laser[path_laser<0]=0.
        path_laser[path_laser > D] = D
        path = path_laser + path_dectect
        T = (1 - np.exp(-c_coef * path))
        return T * dV

    def QC_count(self,
                 thresh=0.5,
                 drop=True):
        """
        Mask angles with too few valid sets.

        Parameters
        ----------
        thresh : float, optional
            Minimum fraction of valid (non-NaN) sets. Default 0.5.
        drop : bool, optional
            If True (default), drop masked angles instead of setting them to NaN.

        Returns
        -------
        xarray.DataArray
            Filtered ``P11``.
        """
        QC_set_count = self.P11.count('set') / len(self.P11.set) > thresh
        return self.P11.where(QC_set_count, drop=drop)

    def QC_CV(self,
              thresh=0.5,
              drop=True):
        """
        Mask angles with a high coefficient of variation between sets.

        Parameters
        ----------
        thresh : float, optional
            Maximum coefficient of variation (std / mean over sets). Default 0.5.
        drop : bool, optional
            If True (default), drop masked angles instead of setting them to NaN.

        Returns
        -------
        xarray.DataArray
            Filtered ``P11``.
        """
        QC_set_CV = (self.P11.std('set') / self.P11.mean('set')) < thresh
        return self.P11.where(QC_set_CV, drop=drop)

    def apply_QC(self,
                 drop=True):
        """
        Apply quality control to ``P11`` with :meth:`QC_count` and :meth:`QC_CV`.

        Parameters
        ----------
        drop : bool, optional
            If True (default), drop masked angles instead of setting them to NaN.
        """
        self.P11 = self.QC_count(drop=drop)
        self.P11 = self.QC_CV(drop=drop)

    def full_process(self,
                     quality_control=True,
                     drop=False):
        """
        Run the full processing chain.

        Calls :meth:`auxdata`, :meth:`get_attenuation`, :meth:`get_angles`,
        :meth:`get_scattering_volume`, :meth:`process_large_angles`,
        :meth:`process_forward_angles`, :meth:`get_matrix_terms`,
        :meth:`merge_angles` and optionally :meth:`apply_QC`.

        Parameters
        ----------
        quality_control : bool, optional
            If True (default), apply quality control on ``P11``.
        drop : bool, optional
            If True, drop angles rejected by quality control instead of
            setting them to NaN. Default False.
        """
        self.auxdata()

        self.get_attenuation()
        self.get_angles()
        self.get_scattering_volume()
        self.process_large_angles()
        self.process_forward_angles()
        self.get_matrix_terms()
        self.merge_angles()
        if quality_control:
            self.apply_QC(drop=drop)
