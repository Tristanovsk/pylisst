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
Definitions of the optical quantities are given in :doc:`theory`.

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
measured in clean water (subscript :math:`w`, median per PMT gain) to
compensate for laser drift. The beam attenuation coefficient
(:eq:`eq-c`) follows from the Beer-Lambert law:

.. math::
   :label: eq-tau

   \tau = \frac{LP / LREF}{\left(LP_w / LREF_w\right)}, \qquad
   c = -\frac{\ln \tau}{L}

with :math:`L = 0.15` m the optical path of the sample chamber. :math:`\tau` and
:math:`c` are computed for each laser polarization (``config``).

Large angles (eyeball)
----------------------

Eyeball signals are noted :math:`s_{dl}`, with :math:`d \in \{r, p\}` the
detector and :math:`l \in \{p, r\}` the laser polarization (first or second
rotation): :math:`r_p`, :math:`r_r`, :math:`p_p` and :math:`p_r`. After correction of
the half-wave plate transmission (applied once to the second laser
reference and to :math:`r_p` and :math:`r_r`, see
:meth:`~pylisst.process.process.correct_half_wave_plate`), each signal is corrected
(:meth:`~pylisst.process.process.process_large_angles`) as

.. math::
   :label: eq-eyeball

   s_{dl}^{corr}(\theta) = G(\theta)\,\sin\theta\;
   \frac{s_{dl}(\theta) - \dfrac{LREF}{LREF_w}\, s_{dl,w}(\theta)}
        {\exp\left[-c\,\ell(\theta)\right]}

where the clean-water background :math:`s_{dl,w}` is scaled by the laser
reference ratio, and :math:`\ell(\theta)` is the path length along the beam to
the sample volume and from the sample volume to the eyeball,

.. math::
   :label: eq-eyeball-path

   \ell(\theta) = L_e - R\cot\theta + \frac{R}{\sin\theta}

with :math:`L_e = 0.10` m the distance from the transmit window to the eyeball
axis and :math:`R = 0.02` m the distance from the eyeball to the beam;
:math:`\sin\theta`
corrects for the lengthening of the scattering volume seen by the eyeball,
and :math:`G(\theta)` is the geometric calibration polynomial
(:meth:`~pylisst.process.process.get_scattering_volume`). Signals of the first
40 angles are also corrected for a laser power change. The ``r`` detector
signals are finally multiplied by the relative gain :math:`\alpha` of the two
photomultipliers, estimated from data where both detectors see the same
signal unless provided (:meth:`~pylisst.process.process.get_alpha`):

.. math::
   :label: eq-alpha

   \alpha = \operatorname{median}\left(\frac{r_p}{r_r}, \frac{p_p}{p_r}\right)
   _{\theta = 45^\circ, 135^\circ}

Mueller matrix terms (:eq:`eq-mueller`) are then computed by
:meth:`~pylisst.process.process.get_matrix_terms`:

.. math::
   :label: eq-p11-p12

   P_{11} = \frac{1}{4}(r_p + p_p + r_r + p_r), \qquad
   \frac{P_{12}}{P_{11}} = \frac{(p_p - r_p) + (p_r - r_r)}{4 P_{11}}

:math:`P_{22}/P_{11}` is retrieved independently from each laser polarization:

.. math::
   :label: eq-p22

   \left(\frac{P_{22}}{P_{11}}\right)_{1} &=
   \frac{2P_{11} - \left[r_p(1 + \cos 2\theta) + p_p(1 - \cos 2\theta)\right]}
        {2 \cos^2 2\theta \; P_{11}} \\
   \left(\frac{P_{22}}{P_{11}}\right)_{2} &=
   \frac{2P_{11} - \left[r_r(1 - \cos 2\theta) + p_r(1 + \cos 2\theta)\right]}
        {2 \cos^2 2\theta \; P_{11}}

:math:`P_{22}/P_{11}` is the mean of both estimates. Since :math:`\cos 2\theta` vanishes at 45 and
135 degrees, :math:`P_{22}` is masked within 40-50 and 130-140 degrees. For
homogeneous spheres, :math:`P_{22}/P_{11} = 1` (see :doc:`theory`).

Near-forward angles (rings)
---------------------------

The 32 ring detectors have radii :math:`\rho_i` increasing logarithmically
from 0.1 to 20 mm. Their edge angles in water follow from refraction at the
window (:math:`n_w = 1.334`),

.. math::
   :label: eq-ring-angles

   \theta_i = \arcsin\left[\frac{\sin\left(\arctan(\rho_i / f)\right)}{n_w}\right],
   \qquad f = 53\ \text{mm}

and, since each ring covers one sixth of a circle, their solid angles are
:math:`\Delta\Omega_i = \frac{2\pi}{6}(\cos\theta_i - \cos\theta_{i+1})`.
Ring counts :math:`N_i` are corrected for attenuation, background, ring area
(:math:`d_{cal}`), vignetting (:math:`d_{vig}`) and neutral density filter
(:math:`ND`), converted to power and divided by the incident laser power
:math:`P_0` (from ``LREF``)
(:meth:`~pylisst.process.process.process_forward_angles`):

.. math::
   :label: eq-ring-vsf

   \beta(\bar{\theta}_i) =
   \frac{w_r \left(N_i/\tau - \dfrac{LREF}{LREF_w} N_{i,w}\right) d_{cal,i}\, d_{vig,i}\, ND}
        {L\, \Delta\Omega_i\, P_0}

with :math:`w_r` the power per count of the rings and :math:`\bar{\theta}_i` the
ring center angle (:math:`\theta_i\,\rho^{1/2}` with :math:`\rho = 200^{1/32}`, about the
geometric mean of the ring edge angles). The ring VSF is the average over the
two laser polarizations.

Merging and quality control
---------------------------

Eyeball :math:`P_{11}` is relative; it is scaled onto the absolute ring VSF
with the median ratio over the overlapping angles :math:`\Theta_{ov}` (14-15 degrees),

.. math::
   :label: eq-merge

   P_{11}(\theta) \leftarrow
   \operatorname{median}_{\theta' \in \Theta_{ov}}
   \left[\frac{\beta_{ring}(\theta')}{P_{11}(\theta')}\right] P_{11}(\theta)

and both are merged into ``P11`` over the full angular range
(:meth:`~pylisst.process.process.merge_angles`). The scattering and
backscattering coefficients and the phase function are then obtained from
:eq:`eq-b` and :eq:`eq-bb-ratio`. Quality control removes
angles with too few valid sets or a high coefficient of variation between
sets (:meth:`~pylisst.process.process.apply_QC`).

LISST-200X
----------

LISST-200X ``.RBN`` files are read and converted to (uncalibrated) VSF and
beam attenuation by :func:`~pylisst.lisst_x.lisst_200X`.
