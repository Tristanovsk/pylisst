import os
import numpy as np
from scipy.interpolate import interp1d

import xarray as xr
import datetime as dt


class driver:
    def __init__(self, file):
        self.proc_date = dt.datetime.utcnow()
        self.file = file
        self.Nangles = 150
        self.precord = 40 + self.Nangles * 5  # partial record: one polarization scan of 3
        self.record = 2 * self.precord  # 2 turns per set

    def read(self):
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
        return (self.rp_on > 30000) | (self.rr_on > 30000) \
               | (self.pr_on > 30000) | (self.pp_on > 30000)

    def date_parser(self, date):
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
        return xr.DataArray(arr, dims=dims,
                            coords=coords,
                            attrs=dict(
                                description="LISST-VSF",
                                units="-"),
                            name=name
                            ).assign_coords({'pmt': self.pmt_gain})

    def xarray_converter_1d(self, arr, name=""):
        return xr.DataArray(arr, dims=["set"],
                            coords=dict(set=range(self.nsets)),
                            attrs=dict(
                                description="LISST-VSF",
                                units="-"),
                            name=name
                            )

    def reader(self):
        self.read()
        self.preallocate()
        self.parser()
        self.angular_interp()
