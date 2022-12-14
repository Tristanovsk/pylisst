import numpy as np
import xarray as xr

# TODO convert into yaml
class calib:
    def __init__(self):
        self.number_rings = 32
        self.SN = 1660
        self.depth_slope = 0.01
        self.depth_offset = 0
        self.temp_slope = 0.01
        self.temp_offset = 0
        self.bat_slope = 0.01
        self.bat_offset = 0
        self.angle_offset = 5
        self.Watt_per_count_on_rings = 1.9e-10
        self.Watt_per_count_laser_ref = 4.425e-6 * 1.04  # multiplied by 1.04 for Fresnel loss
        self.ND = 10.23
        self.laser_power_change_factor = 14.3
        self.HWPlate_transmission = 0.68
        self.geometric_cal_coeff = np.array([2.78986720465452e-08, -9.74355402574768e-06, 0.00132025877173096,
                                             -0.0834514978333943, 2.47185673759084])

        # Geometric dcal
        self.dcal = np.array(
            [1.01790833330000, 0.992134894300000, 1.01081606840000, 0.994928828300000, 1.00437070280000,
             0.998918404700000, 0.998590554600000, 1.00420485260000, 1.00007630710000, 0.998899970300000,
             1.00094967890000, 1.00040191990000, 1.00111296400000, 1.00046768960000, 1.02135541230000,
             0.999902616600000, 1.11156301020000, 1.12066675740000, 1.24936993590000, 1.16431994250000,
             1.33556567670000, 1.20908922660000, 1.07815400160000, 1.76207523180000, 1.55085634300000,
             2.53041187530000, 2.56385922160000, 3.57572117650000, 3.96319866420000, 5.01664107560000,
             5.77011140700000, 7.01385046680000])
        self.dcal = xr.DataArray(self.dcal, dims=["number"],
                            coords=dict(number=range(self.number_rings)),
                            attrs=dict(description="LISST-VSF calibration coef. for forward angles",
                                       units="-"),)

        # De - vignetting factors
        self.dvig = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.02101205238326,
             1.20581086480367, 1.43697017345649, 1.69141875258289, 2.00024365577338, 2.44936150814052])
        self.dvig = xr.DataArray(self.dvig, dims=["number"],
                            coords=dict(number=range(self.number_rings)),
                            attrs=dict(description="LISST-VSF vignetting factors for forward angles",
                                       units="-"))