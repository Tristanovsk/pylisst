Usage
=====

LISST-VSF
---------

A LISST-VSF processing requires a sample measurement and a clean-water
background (``Z``) measurement, both read with :class:`pylisst.driver.driver`:

.. code-block:: python

   from pylisst import driver, calib, process

   scat = driver('V1111510.VSF')   # sample
   scat.reader()
   zsc = driver('Z1110820.VSF')    # clean-water background
   zsc.reader()

   p = process(scat, zsc, calib())
   p.full_process()

Main outputs of :class:`pylisst.process.process`:

============ ==========================================================
Attribute    Description
============ ==========================================================
``beam_c``   Beam attenuation coefficient (m-1)
``ring_vsf`` VSF from the ring detectors (m-1 sr-1)
``p11``      Eyeball P11 (VSF)
``p12``      Eyeball P12 normalized by P11
``p22``      Eyeball P22 normalized by P11
``P11``      VSF merged from rings and eyeball over the full angular range
``alpha``    Relative gain of the two eyeball photomultipliers
============ ==========================================================

All outputs are :class:`xarray.DataArray` objects with a ``set``
dimension (one per measurement) and, for angular quantities, an
``angles`` dimension in degrees.

Plotting the VSF over both near-forward and large angles:

.. code-block:: python

   import matplotlib.pyplot as plt
   from pylisst.utils import plot

   fig, ax = plt.subplots()
   ax, axlin = plot().semilog(ax)
   for a in (ax, axlin):
       p.P11.median('set').plot(ax=a, color='black')

LISST-200X
----------

.. code-block:: python

   from pylisst import lisst_200X

   ds = lisst_200X('sample.rbn')
   ds.vsf.plot(hue='set')
