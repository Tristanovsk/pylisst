"""
PyLisst: scientific code to process LISST-VSF measurements.

The package reads raw binary files of the Sequoia LISST-VSF instrument,
applies background (clean-water) and attenuation corrections and computes
the volume scattering function (VSF) together with the normalized Mueller
matrix terms (P11, P12, P22) over near-forward (rings) and large (eyeball)
scattering angles. A reader for LISST-200X ``.RBN`` files is also provided.

Main entry points
-----------------
:class:`~pylisst.driver.driver`
    Reader/parser of raw LISST-VSF ``.VSF`` binary files.
:class:`~pylisst.calibration.calib`
    Instrument calibration factors.
:class:`~pylisst.process.process`
    Full processing chain from raw counts to VSF and Mueller matrix terms.
:func:`~pylisst.lisst_x.lisst_200X`
    Reader and processor of LISST-200X ``.RBN`` files.

Examples
--------
>>> from pylisst import driver, calib, process
>>> scat = driver('V1111510.VSF')   # sample measurement
>>> scat.reader()
>>> zsc = driver('Z1110820.VSF')    # clean-water background
>>> zsc.reader()
>>> p = process(scat, zsc, calib())
>>> p.full_process()
>>> p.P11  # VSF over the full angular range

Version history
---------------
- v1.0.1: remove background Z measurement before attenuation correction
- v1.0.2: add reader and process for LISST-200X
"""

__version__='1.0.2'

from . import utils
from .driver import driver
from .calibration import calib
from .process import process
from .lisst_x import lisst_200X