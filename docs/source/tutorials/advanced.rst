Advanced
========

Complete processing workflows of the HABLAB experiment, in which LISST-VSF
measurements were acquired on cultures of phytoplankton species, together with
calibrated polystyrene beads and measurements of inherent optical properties (IOPs).

.. mermaid::

   flowchart LR
      RAW[Raw .VSF files<br/>+ sample sheet] --> BEADS[beads_process<br/>validation with Mie]
      RAW --> HAB[hablab_process<br/>batch processing]
      HAB --> L2[Level-2 netCDF<br/>beam_c, P11, p12, p22]
      L2 --> POST[post_processing<br/>phase function, b<sub>b</sub>/b]
      IOP[IOP measurements<br/>b, a, Hyper-bb] --> POST

* :doc:`beads_process` -- validation of the processing with polystyrene beads of
  several sizes: standard vs. modified scattering-volume correction, comparison of
  the Mueller matrix terms and of the backscattering ratio with Mie theory.
* :doc:`hablab_process` -- batch processing of all samples listed in a sample sheet
  into Level-2 netCDF files (one per sample).
* :doc:`post_processing` -- analysis of the Level-2 files with the IOPs measured on
  the same samples: phase functions, fit with analytical models (Henyey-Greenstein,
  TTRM), backscattering ratio, mean cosine and :math:`\chi(117^\circ)` factor, and
  comparison with Hyper-bb measurements.

.. note::

   These notebooks rely on data stored locally and on personal packages
   (``mie_perso`` for Mie computations, ``pffit`` for phase function fitting,
   ``invRrs`` for water optical properties and reflectance modeling). They are
   rendered with their saved outputs and are not executed when building the
   documentation.

.. toctree::
   :maxdepth: 2
   :caption: Advanced
   :hidden:

   beads_process
   hablab_process
   post_processing
