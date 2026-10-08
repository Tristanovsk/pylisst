Processing chain
================

A LISST-VSF processing combines a sample measurement and a clean-water
background (``Z`` file) measurement. Both raw ``.VSF`` files are read with
:class:`~pylisst.driver.driver`, then processed with
:class:`~pylisst.process.process` and the calibration factors of
:class:`~pylisst.calibration.calib`.

.. mermaid::

   flowchart TD
      S[Sample .VSF file] --> RS[driver.reader]
      Z[Clean-water Z .VSF file] --> RZ[driver.reader]
      RS --> A[auxdata<br/>depth, temperature, battery]
      RS --> ATT
      RZ --> ATT[get_attenuation<br/>tau, beam_c]
      CAL[calib] --> ATT
      ATT --> ANG[get_angles<br/>get_scattering_volume]
      ANG --> LA[process_large_angles<br/>eyeball rp, rr, pp, pr]
      ANG --> FA[process_forward_angles<br/>ring_vsf]
      LA --> MT[get_matrix_terms<br/>p11, p12, p22]
      MT --> MG[merge_angles<br/>P11 over full angular range]
      FA --> MG
      MG --> QC[apply_QC]

The whole chain is run with :meth:`~pylisst.process.process.full_process`.

Reading raw data
----------------

Each measurement set holds two eyeball rotations: the first one with the
laser polarized perpendicular to the scattering plane, the second one with
the laser polarized parallel (via a half-wave plate). For each rotation,
two detectors (``r`` and ``p``) are recorded with laser on and off, along
with the 32 ring detectors and auxiliary data.
:meth:`~pylisst.driver.driver.reader` subtracts the dark (laser off) signal
and interpolates eyeball signals onto a common 1-degree angular grid.

Beam attenuation
----------------

The transmittance :math:`\tau` is the ratio of transmitted laser power
(``LP``) to laser reference (``LREF``), normalized by the same ratio
measured in clean water to compensate for laser drift. The beam attenuation
coefficient is

.. math::

   c = -\frac{\ln \tau}{L}

with :math:`L = 0.15` m the optical path of the sample chamber.

Large angles (eyeball)
----------------------

Eyeball signals are corrected for the half-wave plate transmission, the
clean-water background (scaled by the laser reference ratio), attenuation
along the beam and from the sample volume to the eyeball, scattering volume
lengthening (:math:`\sin\theta`), laser power change over the first angles
and geometry. The relative gain :math:`\alpha` of the two photomultipliers
is estimated from data at 45 and 135 degrees
(:meth:`~pylisst.process.process.get_alpha`) unless provided.

Mueller matrix terms are then computed by
:meth:`~pylisst.process.process.get_matrix_terms`:

.. math::

   P_{11} = \frac{1}{4}(r_p + p_p + r_r + p_r), \qquad
   P_{12} = \frac{(p_p - r_p) + (p_r - r_r)}{4 P_{11}}

P22 is the mean of two estimates obtained from each laser polarization.

Near-forward angles (rings)
---------------------------

Ring counts are corrected for attenuation, background, ring area,
vignetting and neutral density filter, and converted to VSF using the ring
solid angles in water and the incident laser power
(:meth:`~pylisst.process.process.process_forward_angles`).

Merging and quality control
---------------------------

Eyeball P11 is scaled onto the ring VSF over the overlapping angles and
both are merged into ``P11`` over the full angular range
(:meth:`~pylisst.process.process.merge_angles`). Quality control removes
angles with too few valid sets or a high coefficient of variation between
sets (:meth:`~pylisst.process.process.apply_QC`).

LISST-200X
----------

LISST-200X ``.RBN`` files are read and converted to (uncalibrated) VSF and
beam attenuation by :func:`~pylisst.lisst_x.lisst_200X`.
