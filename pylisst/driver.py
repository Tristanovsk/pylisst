"""
Reader of raw LISST-VSF binary files.
"""

import os
import numpy as np
from scipy.interpolate import interp1d

import xarray as xr
import datetime as dt


class driver:
    """
    Reader and parser of a raw LISST-VSF ``.VSF`` binary file.

    Each measurement set holds two eyeball rotations: the first one with the
    laser polarized perpendicular to the scattering plane, the second one
    with the laser polarized parallel (via a half-wave plate). For each
    rotation, the eyeball records two signals (``r`` and ``p`` detectors)
    with laser on and off, along with the 32 ring detectors and auxiliary
    data (laser power, laser reference, depth, temperature, time).

    Parameters
    ----------
    file : str
        Path to the raw ``.VSF`` binary file.

    Attributes
    ----------
    file : str
        Path to the raw file.
    proc_date : datetime.datetime
        Processing date (UTC).
    Nangles : int
        Number of eyeball angles per rotation (150).
    precord : int
        Number of 16-bit words in one eyeball rotation.
    record : int
        Number of 16-bit words in one measurement set (two rotations).

    Notes
    -----
    After :meth:`reader`, the main outputs are the dark-corrected eyeball
    signals interpolated on a common 1-degree angular grid
    (:class:`xarray.DataArray` with dimensions ``(set, angles)``):

    - ``rp``, ``rr``: perpendicular laser polarization, ``p`` and ``r`` detectors;
    - ``pp``, ``pr``: parallel laser polarization, ``p`` and ``r`` detectors;

    together with ``LP`` (transmitted laser power), ``LREF`` (laser reference),
    ``rings1``/``rings2`` (ring detector counts), ``pmt_gain``, ``depth``,
    ``tempC``, ``time`` and the saturation flag ``qc_saturated``.

    Examples
    --------
    >>> scat = driver('V1111510.VSF')
    >>> scat.reader()
    >>> scat.rp.plot(hue='set')
    """

    def __init__(self, file):
        self.proc_date = dt.datetime.utcnow()
        self.file = file
        self.Nangles = 150
        self.precord = 40 + self.Nangles * 5  # partial record: one polarization scan of 3
        self.record = 2 * self.precord  # 2 turns per set

    def read(self):
        """
        Load the raw binary file into an integer array.

        The file is read as big-endian 16-bit words, both unsigned (header,
        rings, auxiliary data) and signed (eyeball data, depth, temperature),
        and reshaped into ``(nsets, record)``.

        Sets attributes ``raw``, ``nsets``, ``batt_volts``, ``pmt_gain``,
        ``pow_trns`` (transmitted power) and ``pow_lref`` (laser reference).
        """
        file = self.file
        # open raw data
        fid = open(file, "rb")
        raw1 = np.fromfile(fid, dtype='>u2')

        nsets = int(len(raw1) / self.record)  # number of sets of 2-turns per set
        if nsets >= 2:
            raw1 = np.reshape(raw1, (nsets, self.record))  # .T
        # read in signed eyeball data
        fid = open(file, "rb")
        raw2 = np.fromfile(fid, dtype='>i2')
        fid.close()
        if nsets >= 2:
            raw2 = np.reshape(raw2, (nsets, self.record))  # .T

        # reshape raw data
        raw = np.zeros(raw2.shape, dtype=object)
        raw[:, :40] = raw1[:, :40]
        raw[:, 790:831] = raw1[:, 790:831]
        raw[:, 40:790] = raw2[:, 40:790]
        raw[:, 831:1580] = raw2[:, 831:1580]

        # depth and temperature are signed values
        raw[:, [36, 37, 826, 827]] = raw2[:, [36, 37, 826, 827]]
        raw = raw.astype(int)
        self.raw = raw
        self.nsets = nsets
        self.batt_volts = raw[:, 33]
        # this is common to the entire set; yet allows different PMT settings for the 2 PMT''s.
        self.pmt_gain = self.xarray_converter_1d(raw[:, 34], name='pmt')
        self.pow_trns = self.xarray_converter_1d(raw[:, 822], name='transmission')
        self.pow_lref = self.xarray_converter_1d(raw[:, 825], name='Lref')
        return

    def preallocate(self):
        """
        Allocate the arrays filled by :meth:`parser`.
        """

        self.rp_off = np.zeros([self.nsets, self.Nangles])
        self.rp_on = np.zeros([self.nsets, self.Nangles])

        self.pp_on = np.zeros([self.nsets, self.Nangles])
        self.pp_off = np.zeros([self.nsets, self.Nangles])

        self.pr_off = np.zeros([self.nsets, self.Nangles])
        self.pr_on = np.zeros([self.nsets, self.Nangles])

        self.rr_off = np.zeros([self.nsets, self.Nangles])
        self.rr_on = np.zeros([self.nsets, self.Nangles])

        self.angles1 = np.zeros([self.nsets, self.Nangles])
        self.angles2 = np.zeros([self.nsets, self.Nangles])
        self.rings1 = np.zeros([self.nsets, 32])
        self.rings2 = np.zeros([self.nsets, 32])
        self.lp1 = np.zeros([self.nsets])
        self.lp2 = np.zeros([self.nsets])
        self.lref1 = np.zeros([self.nsets])
        self.lref2 = np.zeros([self.nsets])
        self.depth1 = np.zeros([self.nsets])
        self.depth2 = np.zeros([self.nsets])
        self.temp1 = np.zeros([self.nsets])
        self.temp2 = np.zeros([self.nsets])
        self.date1 = np.zeros([self.nsets, 2])
        self.date2 = np.zeros([self.nsets, 2])

    def parser(self):
        """
        Split raw records into ring, eyeball and auxiliary variables.

        For each set, extracts the ring counts, laser power, laser reference,
        depth, temperature, date and the eyeball angles and signals
        (laser on/off) of both rotations. Sets the combined attributes
        ``LP``, ``LREF``, ``depth``, ``tempC`` and ``time`` with shape
        ``(nsets, 2)`` (one column per rotation).
        """
        for i in range(self.nsets):
            raw = self.raw
            # --------------------------------------------------
            # First eyeball rotation is polarized perpendicular
            # --------------------------------------------------
            ii = 0
            ie = ii + self.precord
            self.rings1[i, :] = raw[i, ii:ii + 32]
            self.lp1[i] = raw[i, ii + 32]
            self.lref1[i] = raw[i, ii + 35]
            self.depth1[i] = raw[i, ii + 36]
            self.temp1[i] = raw[i, ii + 37]
            self.date1[i, :] = [raw[i, ii + 38], raw[i, ii + 39]]
            self.angles1[i, :] = raw[i, ii + 40:ie:5]

            self.rp_on[i, :] = raw[i, ii + 41:ie:5]
            self.rp_off[i, :] = raw[i, ii + 42:ie:5]
            self.rr_on[i, :] = raw[i, ii + 43:ie:5]
            self.rr_off[i, :] = raw[i, ii + 44:ie:5]

            # -------------------------------------------------
            # Second eyeball rotation is polarized parallel
            # -------------------------------------------------
            ii = ii + self.precord
            ie = ie + self.precord
            self.rings2[i, :] = raw[i, ii:ii + 32]
            self.lp2[i] = raw[i, ii + 32]
            self.lref2[i] = raw[i, ii + 35]
            self.depth2[i] = raw[i, ii + 36]
            self.temp2[i] = raw[i, ii + 37]
            self.date2[i, :] = [raw[i, ii + 38], raw[i, ii + 39]]
            self.angles2[i, :] = raw[i, ii + 40:ie:5]

            self.pp_on[i, :] = raw[i, ii + 41:ie:5]
            self.pp_off[i, :] = raw[i, ii + 42:ie:5]
            self.pr_on[i, :] = raw[i, ii + 43:ie:5]
            self.pr_off[i, :] = raw[i, ii + 44:ie:5]

        self.date1 = self.date_parser(self.date1)
        self.date2 = self.date_parser(self.date2)

        self.LP = np.array([self.lp1, self.lp2]).T
        self.LREF = np.array([self.lref1, self.lref2]).T

        self.depth = np.array([self.depth1, self.depth2]).T
        self.tempC = np.array([self.temp1, self.temp2]).T
        self.time = np.array([self.date1, self.date2]).T

    def angular_interp(self):
        """
        Interpolate eyeball signals onto a common angular grid.

        Eyeball encoder positions differ slightly between sets and rotations;
        signals are linearly interpolated (ignoring zero values) onto a
        1-degree grid spanning all sets. Dark signal (laser off) is then
        subtracted and results are converted to :class:`xarray.DataArray`
        (attributes ``rp``, ``rr``, ``pp``, ``pr``, ``LP``, ``LREF``,
        ``rings1``, ``rings2`` and ``qc_saturated``).
        """
        angle_min = np.min([*self.angles1[:, 0], *self.angles2[:, 0]])
        angle_max = np.max([*self.angles1[:, -1], *self.angles2[:, -1]])
        # set increment in angles
        step = 1
        self.angles = np.arange(angle_min, angle_max + step, step)
        self.angles_idx = self.angles

        def finterp(x, y, x_):
            x = x[y != 0]
            y = y[y != 0]
            return interp1d(x, y, kind='linear', axis=0)(x_)

        # loop to reproject on common angles
        for i in range(self.nsets):
            self.rp_on[i] = finterp(self.angles1[i], self.rp_on[i], self.angles)
            self.rp_off[i] = finterp(self.angles1[i], self.rp_off[i], self.angles)
            self.rr_on[i] = finterp(self.angles1[i], self.rr_on[i], self.angles)
            self.rr_off[i] = finterp(self.angles1[i], self.rr_off[i], self.angles)

            self.pp_on[i] = finterp(self.angles1[i], self.pp_on[i], self.angles)
            self.pp_off[i] = finterp(self.angles1[i], self.pp_off[i], self.angles)
            self.pr_on[i] = finterp(self.angles1[i], self.pr_on[i], self.angles)
            self.pr_off[i] = finterp(self.angles1[i], self.pr_off[i], self.angles)

        # convert into xarray for further processing
        self.rp = self.xarray_converter(self.rp_on - self.rp_off, dims=["set", "angles"],
                                        coords=dict(set=range(self.nsets), angles=self.angles))
        self.rr = self.xarray_converter(self.rr_on - self.rr_off, dims=["set", "angles"],
                                        coords=dict(set=range(self.nsets), angles=self.angles))
        self.pp = self.xarray_converter(self.pp_on - self.pp_off, dims=["set", "angles"],
                                        coords=dict(set=range(self.nsets), angles=self.angles))
        self.pr = self.xarray_converter(self.pr_on - self.pr_off, dims=["set", "angles"],
                                        coords=dict(set=range(self.nsets), angles=self.angles))

        self.LP = self.xarray_converter(self.LP, dims=["set", "config"],
                                        coords=dict(set=range(self.nsets), config=range(2)))
        self.LREF = self.xarray_converter(self.LREF, dims=["set", "config"],
                                          coords=dict(set=range(self.nsets), config=range(2)))
        self.rings1 = self.xarray_converter(self.rings1, dims=["set", "number"],
                                            coords=dict(set=range(self.nsets), number=range(32)))
        self.rings2 = self.xarray_converter(self.rings1, dims=["set", "number"],
                                            coords=dict(set=range(self.nsets), number=range(32)))
        self.qc_saturated = self.xarray_converter(self.mask_saturated(), dims=["set", "angles"],
                                                  coords=dict(set=range(self.nsets), angles=self.angles))

    def mask_saturated(self):
        """
        Flag saturated eyeball measurements.

        Returns
        -------
        numpy.ndarray of bool
            True where any of the laser-on eyeball signals exceeds 30000 counts,
            shape ``(nsets, Nangles)``.
        """
        return (self.rp_on > 30000) | (self.rr_on > 30000) \
               | (self.pr_on > 30000) | (self.pp_on > 30000)

    def date_parser(self, date):
        """
        Convert instrument date words into timestamps.

        Parameters
        ----------
        date : numpy.ndarray
            Array of shape ``(nsets, 2)`` with encoded ``DDDHH`` (day of year
            and hour) and ``MMSS`` (minute and second) values.

        Returns
        -------
        numpy.ndarray of datetime64[us]
            Timestamps of each set.

        Warnings
        --------
        The year is not stored in the raw file and is currently hard-coded to 2022.
        """
        MM = (np.fix(date[:, 1] / 100)).astype(int)
        SS = (date[:, 1] - 100 * MM).astype(int)
        DD = (np.fix(date[:, 0] / 100)).astype(int)
        HH = (date[:, 0] - 100 * DD).astype(int)
        time = np.empty(self.nsets, dtype='datetime64[us]')
        for i in range(self.nsets):
            time[i] = dt.datetime.strptime(
                str(2022) + "-" + str(DD[i]) + ' ' + str(HH[i]) + ':' + str(MM[i]) + ':' + str(SS[i]), "%Y-%j %H:%M:%S")
        return time

    def xarray_converter(self, arr, dims, coords, name=""):
        """
        Wrap an array into a :class:`xarray.DataArray` with a ``pmt`` coordinate.

        Parameters
        ----------
        arr : array_like
            Data to wrap.
        dims : list of str
            Dimension names (first one must be ``set``).
        coords : dict
            Coordinates of the dimensions.
        name : str, optional
            Name of the DataArray.

        Returns
        -------
        xarray.DataArray
        """
        return xr.DataArray(arr, dims=dims,
                            coords=coords,
                            attrs=dict(
                                description="LISST-VSF",
                                units="-"),
                            name=name
                            ).assign_coords({'pmt': self.pmt_gain})

    def xarray_converter_1d(self, arr, name=""):
        """
        Wrap a per-set 1-D array into a :class:`xarray.DataArray` of dimension ``set``.

        Parameters
        ----------
        arr : array_like
            Data of length ``nsets``.
        name : str, optional
            Name of the DataArray.

        Returns
        -------
        xarray.DataArray
        """
        return xr.DataArray(arr, dims=["set"],
                            coords=dict(set=range(self.nsets)),
                            attrs=dict(
                                description="LISST-VSF",
                                units="-"),
                            name=name
                            )

    def reader(self):
        """
        Run the full reading chain.

        Calls :meth:`read`, :meth:`preallocate`, :meth:`parser` and
        :meth:`angular_interp` in sequence.
        """
        self.read()
        self.preallocate()
        self.parser()
        self.angular_interp()
