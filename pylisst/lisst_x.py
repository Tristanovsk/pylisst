"""
Reader of LISST-200X ``.RBN`` binary files.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

import datetime as dt
import matplotlib.pyplot as plt


opj = os.path.join


def lisst_200X(datafile):
    """
    Read and process a LISST-200X ``.RBN`` binary file into a VSF.

    Python port of the Sequoia MATLAB reader. Records are identified by
    their ID (data, background, ring area, configuration, housekeeping,
    volume/area conversion vectors). Ring data are corrected for the
    background, attenuation and ring area, then converted to VSF using the
    ring solid angles in water.

    Parameters
    ----------
    datafile : str
        Path to the ``.RBN`` file.

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions ``set`` (measurement) and ``scat_ang``
        (ring center angle in water, degrees) containing:

        - ``vsf``: volume scattering function (magnitude not calibrated,
          the laser reference calibration being unknown), negative values masked;
        - ``c_att``: beam attenuation coefficient (m-1);
        - ``datetime``: timestamp of each set;
        - ``depth``: depth (m);
        - ``temperature``: temperature (degrees Celsius).

        Instrument configuration (serial number, firmware, path lengths,
        sampling settings, ...) is stored in the dataset attributes.

    Notes
    -----
    The first measurement of the file is discarded.
    """
    # record IDs
    DCAL_RECORD_ID = 44778
    TV1_RECORD_ID = 49391
    TV2_RECORD_ID = 49394
    TA1_RECORD_ID = 44047
    TA2_RECORD_ID = 44274
    ZSCAT_RECORD_ID = 47820
    FZSCAT_RECORD_ID = 64428
    CONFIG_RECORD_ID = 19529
    HOUSEK_RECORD_ID = 52928
    DATA_RECORD_ID = 56026

    Nrings = 36

    fid = open(datafile, 'rb')  # open file for reading using big endian format
    fileSize = fid.seek(0, 2)  # *8 # get the file size in bytes

    fid.seek(0, 0)
    recordID = int(np.fromfile(fid, dtype='>u2', count=1))  # read the first record ID
    # calculate number of records in the file
    RecordSize = 120  # 120 bytes per record
    numRecords = fileSize // RecordSize

    # calculate the number of variables in each record
    num16 = (RecordSize // 2) - 1
    num32 = int((RecordSize - 2) / 4)
    zsc, fzs, dcal, config, housek = {}, {}, {}, {}, {}

    raw_data = np.full((numRecords, num16), np.nan)  # preallocate data array for big speed increase

    # loop through the file and read in the data according to the record ID
    for recordNumber in range(1, numRecords):

        if recordID == ZSCAT_RECORD_ID:
            zsc = np.fromfile(fid, dtype='>u2', count=num16)  # fread(fid, num16, 'uint16'))
        elif recordID == DATA_RECORD_ID:
            raw_data[recordNumber - 1] = np.fromfile(fid, dtype='>u2', count=num16)  # fread(fid, num16, 'uint16'))
        elif recordID == DCAL_RECORD_ID:
            dcal_ = np.fromfile(fid, dtype='>u2', count=num16)  # fread(fid, num16, 'uint16'))
            dcal = dcal_[1:Nrings + 1] / dcal_[0]
        elif recordID == FZSCAT_RECORD_ID:
            fzs = np.fromfile(fid, dtype='>u2', count=num16)  # fread(fid, num16, 'uint16'))
        elif recordID == CONFIG_RECORD_ID:

            fid.seek(-2, 1)
            # print(np.fromfile(fid, dtype=np.str_, count=20))
            config['name'] = ''.join(map(chr,np.fromfile(fid, dtype='>u1', count=20)))  # ''.join() #fread(fid, 20, '*char')))
            config['serialNumber'] = np.fromfile(fid, dtype='>u2', count=1)[0]  # fread(fid, 1, 'uint16'))[0]
            config['firmwareVer'] = np.fromfile(fid, dtype='>u2', count=1)[
                                        0] * 0.001  # fread(fid, 1, 'uint16'))[0] * 0.001
            config['VCC'] = np.fromfile(fid, dtype='>u4', count=1)[0]  # fread(fid, 1, 'uint32'))[0]
            # optical path in m
            config['fullPath'] = np.fromfile(fid, dtype='>u2', count=1)[
                                     0] * 0.01 * 1e-3  # fread(fid, 1, 'uint16'))[0] * 0.01
            config['effPath'] = np.fromfile(fid, dtype='>u2', count=1)[
                                    0] * 0.01 * 1e-3  # fread(fid, 1, 'uint16'))[0] * 0.01

            config['bioBlock'] = np.fromfile(fid, dtype='>u1', count=1)[0]  # fread(fid, 1, 'uint8'))[0]
            config['sTube'] = np.fromfile(fid, dtype='>u1', count=1)[0]  # fread(fid, 1, 'uint8'))[0]
            config['analogConcScale'] = np.fromfile(fid, dtype='>u2', count=1)[0]  # fread(fid, 1, 'uint16'))[0]
            config['endcap'] = np.fromfile(fid, dtype='>u2', count=1)[0]  # fread(fid, 1, 'uint16'))[0]
            config['startCond'] = np.fromfile(fid, dtype='>u2', count=1)[0]  # fread(fid, 1, 'uint16'))[0]
            config['startCondData'] = ''.join(map(chr,np.fromfile(fid, dtype='>u1', count=20)))  # ''.join(np.fromfile(fid, dtype=np.unicode_, count=20))#fread(fid, 20, '*char')))
            config['stopCond'] = np.fromfile(fid, dtype='>u2', count=1)[0]  # fread(fid, 1, 'uint16'))[0]
            config['stopCondData'] = ''.join(map(chr,np.fromfile(fid, dtype='>u1', count=20))) # ''.join(np.fromfile(fid, dtype=np.unicode_, count=20))#fread(fid, 20, '*char')))
            config['measurementAve'] = np.fromfile(fid, dtype='>u2', count=1)[0]  # fread(fid, 1, 'uint16'))[0]
            config['sampleInterval'] = np.fromfile(fid, dtype='>u2', count=1)[0]  # fread(fid, 1, 'uint16'))[0]
            config['sampleMode'] = np.fromfile(fid, dtype='>u2', count=1)[0]  # fread(fid, 1, 'uint16'))[0]
            config['burstSamples'] = np.fromfile(fid, dtype='>u2', count=1)[0]
            config['burstInterval'] = np.fromfile(fid, dtype='>u2', count=1)[0]
            config['transmitRaw'] = np.fromfile(fid, dtype='>u2', count=1)[0]
            config['lifetimeSamples'] = np.fromfile(fid, dtype='>u4', count=1)[0]
            config['lifetimeLaserOn'] = np.fromfile(fid, dtype='>u4', count=1)[0]
            config['supportBoard'] = np.fromfile(fid, dtype='>u2', count=1)[0]
            config['ambientLight'] = np.fromfile(fid, dtype='>u2', count=1)[0]
        elif recordID == HOUSEK_RECORD_ID:
            housek = np.fromfile(fid, dtype='>f4', count=num32)  # .reshape(1, num32)
        elif recordID == TV1_RECORD_ID:
            Tv = np.fromfile(fid, dtype='>f4', count=num32)  # .reshape(1, num32)
        elif recordID == TV2_RECORD_ID:
            Tv_ = np.fromfile(fid, dtype='>f4', count=7)
        elif recordID == TA1_RECORD_ID:
            Ta = np.fromfile(fid, dtype='>f4', count=num32)  # .reshape(1, num32)
        elif recordID == TA2_RECORD_ID:
            Ta_ = np.fromfile(fid, dtype='>f4', count=7)
        else:
            print(recordNumber, recordID)
            print('Unrecognized data record ID found in .RBN file')

        fid.seek(recordNumber * RecordSize, 0)  # go the location of the next record ID
        recordID = int(np.fromfile(fid, dtype='>u2', count=1))  # read the next record ID

    fid.close()

    Ta = np.concatenate([Ta,Ta_])
    Tv = np.concatenate([Tv, Tv_])

    # remove NaN rows from data matrix that correspond to header data rows
    raw_data = raw_data[~np.isnan(raw_data).any(axis=1)]

    # remove first measurements
    raw_data = raw_data[1:,:]

    data = raw_data[:,:Nrings]
    auxdata = raw_data[:,Nrings:]


    # # negative ring values are possible, data must be corrected
    # data[data[:,0:36]>40950] = data[data[:,0:36]>40950] - 65536
    # fzs[fzs[:,0:36]>40950] = fzs[fzs[:,0:36]>40950] - 65536
    # zsc[zsc[:,0:36]>40950] = zsc[zsc[:,0:36]>40950] - 65536

    nsets, nextra = data.shape

    data = data / 10
    fzs[:Nrings] = fzs[:Nrings] / 10
    zsc[:Nrings] = zsc[:Nrings] / 10

    Lref_zsc = zsc[Nrings + 3]
    LaserRatio = zsc[Nrings] / Lref_zsc  # ratio of transmitted power / laser ref

    Lref = xr.DataArray(auxdata[:, 3], dims=["set"],
                        coords=dict(set=range(nsets)),
                        name='Lref')

    xzsc = xr.DataArray(zsc[:Nrings], dims=["ring"],
                        coords=dict(ring=range(Nrings)),
                        name='zsc'
                        )

    dcal = xr.DataArray(dcal, dims=["ring"],
                        coords=dict(ring=range(Nrings)),
                        name='dcal'
                        )

    tau = xr.DataArray(auxdata[:, 0] / Lref / LaserRatio, dims=["set"],  # /
                       coords=dict(set=range(nsets)))

    c_att = xr.DataArray(-np.log(tau) / config['effPath'], dims=["set"],  # /
                         coords=dict(set=range(nsets)),
                         name='c_att'
                         )

    scat = xr.DataArray(data, dims=["set", "ring"],
                        coords=dict(set=range(nsets), ring=range(Nrings)),
                        attrs=dict(
                            description="LISST-200X",
                            units="-"),
                        name='scat'
                        )

    # subtract the background
    scat = scat - xzsc * Lref / Lref_zsc

    # correct for attenuation
    scat = scat / tau

    # apply ring area file
    scat = scat * dcal

    # calculate angles in water in radians(120 mm focal length)
    rho = 1.18
    m_water = 1.334
    rings = np.arange(0, 36 + 1)
    theta0air = 0.102 / 120
    edge_angles = theta0air * rho ** rings

    # convert air angles to in water angles
    edge_angles = np.arcsin(np.sin(edge_angles) / m_water)
    # find solid angle
    dOmega = np.cos(edge_angles[:-1]) - np.cos(edge_angles[1:])
    dOmega = dOmega * 2 * np.pi / 6  # factor 6 takes care of rings covering only 1/6th circle

    # calculate detector center angles in degrees
    angles = np.degrees(np.sqrt(edge_angles[:-1] * edge_angles[1:]))

    # compute light on rings
    Watt_per_count_on_rings = 1.9e-10  # assumed the same for all detectors

    light_on_rings = scat * Watt_per_count_on_rings
    # calculate incident laser power from LREF
    Watt_per_count_laser_ref = 1  # THIS IS UNKNOWN

    laser_incident_power = Lref * Watt_per_count_laser_ref

    # compute VSF(no calibration for Lref, so magnitude is not correct)
    vsf = light_on_rings / dOmega / laser_incident_power / config['effPath']
    # mask negative or null values
    vsf=vsf.where(vsf >= 0)

    # # normalize to factory LREF
    # vsf = vsf * fzs[Nrings+3] / Lref
    #
    # # apply concentration  calibration
    # vsf = vsf / config['VCC']

    # get anciliary data
    scale = np.ones(23)
    scale[0] = housek[16]  # Laser Transmission
    scale[1] = 0.01  # Supply Voltage
    scale[2] = 0.0001  # Analog Input 1
    scale[3] = housek[15]  # Laser Reference
    scale[4] = housek[2]  # Depth
    scale[5] = housek[11]  # Temperature
    scale[12] = 0.0001  # Analog Input 2
    scale[13] = housek[14]  # Sauter Mean Diameter
    scale[14] = housek[13]  # Total Volume Concentration
    scale[22] = 0.0001  # Analog Input 3
    offset = np.zeros(23)
    offset[4] = housek[3]  # Depth
    offset[5] = housek[12]  # Temperature

    # apply scaling factors
    auxdata = auxdata * scale+ offset
    scale[14] = 0.01  # total concentration replaced by path length (x100) in background records
    zsc[Nrings:] = zsc[Nrings:] * scale + offset
    fzs[Nrings:] = fzs[Nrings:] * scale + offset

    datetime = pd.DataFrame(auxdata[:, 6:12], columns=('year', 'month', 'day', 'hour', 'minute', 'second'))
    datetime = pd.to_datetime(datetime)
    datetime.index.name='set'
    datetime=datetime.to_xarray()
    datetime.name='datetime'

    depth = xr.DataArray(auxdata[:,4], dims=["set"], name='depth',
                       coords=dict(set=range(nsets)))
    temperature = xr.DataArray(auxdata[:,5], dims=["set"], name='temperature',
                       coords=dict(set=range(nsets)))

    vsf = vsf.assign_coords(scat_ang=('ring', angles)).swap_dims({'ring': 'scat_ang'})
    vsf.name = 'vsf'
    obj = xr.merge([vsf, c_att, datetime,depth, temperature])
    obj.attrs = config

    return obj


def post_process():
    """
    Example comparing LISST-200X VSF with Mie computations for polystyrene beads.

    .. note:: Script-like function with hard-coded paths; requires the
       external ``mie_perso`` package.
    """
    datadir = '/DATA/projet/gernez/hablab/lisst200'
    datafile = opj(datadir, 'pry_c1_r2_s_20220420.rbn')#'b1mic_c1_r1_20220421.rbn')  # 'b3mic_c1_r2_20220421.rbn')
    rn_med = 5.5

    pkg_dir = '/DATA/instrument/lisst-vsf/pylisst'
    from mie_perso import psd, mie_multiprocess

    # ---------------------------------
    # get Mie simulations
    # ---------------------------------
    pmie = mie_multiprocess.processor()
    size_param = psd.size_param
    # psd = psd.psd()

    vsf = lisst_200X(datafile)

    wl = 670
    nMedium = 1.3199 + 6878 / wl ** 2 - 1.132e9 / wl ** 4 + 1.11e14 / wl ** 6
    wl_medium = wl / nMedium
    npolystyrene = {515: 1.60, 670: 1.583}
    m = npolystyrene[wl] / nMedium - 0.000j

    ofile = 'data/mueller_mie_' + format(m, '1.3f') + '_t3600.nc'

    theta = np.linspace(0, np.pi, 3600)
    x = np.logspace(np.log10(1), 2, 1001)
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
    size_param = psd.size_param
    psd_ = psd.psd()

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

    vsf_norm=2*np.pi*vsf.vsf/vsf.c_att * 1e7


    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(16, 5))
    fig.subplots_adjust(bottom=0.15, top=0.925, left=0.1, right=0.975,
                        hspace=0.1, wspace=0.25)

    vsf_norm.plot(hue='set', ax=axs[0],add_legend=False)
    vsf_norm.plot(hue='set', ax=axs[1],add_legend=False)
    axs[0].semilogy()
    axs[1].semilogy()
    axs[1].semilogx()
    axs[0].plot(ang, S11 / norm, c='red', label='Mie')
    axs[1].plot(ang, S11 / norm, c='red', label='Mie')
    axs[0].set_xlim(0,25)

