Basics
======

Getting started with PyLisst: read raw files, run the processing chain and use its
outputs.

* :doc:`quickstart_lisst_vsf` -- read LISST-VSF sample and blank files, inspect the raw
  eyeball signals, run the processing (in one call or step by step), apply quality
  control, plot the VSF and Mueller matrix terms, compute the phase function and the
  backscattering ratio, and export the results to netCDF.
* :doc:`quickstart_lisst_200x` -- read a LISST-200X file into an :class:`xarray.Dataset`.
* :doc:`beads_process_basic` -- a first validation: polystyrene beads compared with
  Mie theory.

.. note::

   The examples use measurements of the HABLAB experiment, which are not distributed
   with the package. The notebooks are rendered with their saved outputs; to run them,
   set the data paths to your own files.

.. toctree::
   :maxdepth: 2
   :caption: Basics
   :hidden:

   quickstart_lisst_vsf
   quickstart_lisst_200x
   beads_process_basic
