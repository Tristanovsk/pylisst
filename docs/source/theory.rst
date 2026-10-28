Theoretical background
======================

This page gathers the definitions used by PyLisst and the Lorenz-Mie theory
used to validate the processing with calibrated polystyrene beads
(see :doc:`tutorials/beads_process_basic` and :doc:`tutorials/beads_process`).
Notations follow Bohren and Huffman (1983) and Sumlin et al. (2018).

Inherent optical properties
---------------------------

A collimated beam crossing a thin layer of a particle suspension is attenuated
by absorption and scattering. The beam attenuation coefficient :math:`c`
(m\ :sup:`-1`) is the sum of the absorption coefficient :math:`a` and the
scattering coefficient :math:`b`:

.. math::
   :label: eq-c

   c = a + b

The angular distribution of the scattered light is described by the volume
scattering function (VSF) :math:`\beta(\theta)` (m\ :sup:`-1` sr\ :sup:`-1`),
the scattered radiant intensity per unit scattering volume and per unit
irradiance, at the scattering angle :math:`\theta` between the incident and
scattered directions. For particles in random orientation, the VSF does not
depend on the azimuth and its integrals over all directions and over the
backward hemisphere give the scattering and backscattering coefficients:

.. math::
   :label: eq-b

   b = 2\pi \int_0^{\pi} \beta(\theta) \sin\theta \, d\theta, \qquad
   b_b = 2\pi \int_{\pi/2}^{\pi} \beta(\theta) \sin\theta \, d\theta

The phase function :math:`\tilde{\beta}(\theta) = \beta(\theta)/b` is then
normalized as

.. math::
   :label: eq-pf-norm

   2\pi \int_0^{\pi} \tilde{\beta}(\theta) \sin\theta \, d\theta = 1

(some notebooks use the equivalent :math:`4\pi`-normalization
:math:`\frac{1}{2}\int_0^{\pi} 4\pi\tilde{\beta}(\theta) \sin\theta \, d\theta = 1`).
Derived quantities are the backscattering ratio :math:`\tilde{b}_b`, the mean
cosine (asymmetry parameter) :math:`g` and the :math:`\chi` factor relating
:math:`b_b` to the VSF at a single backward angle, as used by fixed-angle
backscattering sensors:

.. math::
   :label: eq-bb-ratio

   \tilde{b}_b &= \frac{b_b}{b}
   = \frac{\int_{\pi/2}^{\pi} \beta(\theta) \sin\theta \, d\theta}
          {\int_0^{\pi} \beta(\theta) \sin\theta \, d\theta} \\
   g &= 2\pi \int_0^{\pi} \tilde{\beta}(\theta) \cos\theta \sin\theta \, d\theta \\
   b_b &= 2\pi\,\chi(\theta_0)\,\beta(\theta_0)

with :math:`\theta_0 = 117^\circ` in :doc:`tutorials/post_processing`.

.. note::

   The LISST-VSF measures :math:`\beta(\theta)` over about 0.09-155 degrees.
   Integrals of :eq:`eq-b` restricted to that range underestimate
   :math:`b` and :math:`b_b`; the missing angles are accounted for by fitting an
   analytical phase function, or :math:`b` is taken from an independent
   measurement.

Scattering matrix
-----------------

The polarization state of light is described by the Stokes vector
:math:`\mathbf{I} = (I, Q, U, V)^T`. In the far field of a particle (distance
:math:`r` much larger than the particle size and the wavelength), the Stokes
vectors of the incident and scattered light, both defined with respect to the
scattering plane, are related by a :math:`4\times4` scattering (Mueller)
matrix :math:`\mathbf{S}`:

.. math::
   :label: eq-stokes

   \mathbf{I}_{sca} = \frac{1}{k^2 r^2}\, \mathbf{S}(\theta)\, \mathbf{I}_{inc},
   \qquad k = \frac{2\pi n_{med}}{\lambda}

where :math:`\lambda` is the wavelength in vacuum and :math:`n_{med}` the
refractive index of the host medium (water). For a dilute suspension, the
fields scattered by the particles are incoherent and their intensities add,
so that the scattering of a volume is the sum of the single-particle
contributions (Mie, 1908, §22). The volume scattering matrix
:math:`\mathbf{P}(\theta)` (m\ :sup:`-1` sr\ :sup:`-1`) of particles in random
orientation having a plane of symmetry has six independent elements
(Mishchenko and Travis, 2008):

.. math::
   :label: eq-mueller

   \mathbf{P}(\theta) =
   \begin{pmatrix}
   P_{11} & P_{12} & 0 & 0 \\
   P_{12} & P_{22} & 0 & 0 \\
   0 & 0 & P_{33} & P_{34} \\
   0 & 0 & -P_{34} & P_{44}
   \end{pmatrix}

The first element is the VSF, :math:`P_{11}(\theta) = \beta(\theta)`. For
unpolarized incident light, the degree of linear polarization of the scattered
light is :math:`-P_{12}/P_{11}`. For homogeneous spheres, :math:`P_{22} = P_{11}`
and :math:`P_{44} = P_{33}`, so that a departure of :math:`P_{22}/P_{11}` from 1
is a signature of non-sphericity.

The LISST-VSF measures :math:`P_{11}`, :math:`P_{12}` and :math:`P_{22}`
(see :doc:`processing_chain`). PyLisst returns ``P11`` (the VSF) and the
normalized terms ``p12`` :math:`= P_{12}/P_{11}` and ``p22``
:math:`= P_{22}/P_{11}`.

Lorenz-Mie theory
-----------------

Gustav Mie (1908) derived the exact solution of Maxwell's equations for the
scattering of a plane electromagnetic wave by a homogeneous sphere, in order
to explain the colors of colloidal gold suspensions. A mathematically similar
theory had been developed by Ludvig Lorenz, based on his own theory of light
rather than on Maxwell's electromagnetics, hence the name Lorenz-Mie theory.
The solution expands the incident, internal and scattered fields in series of
vector spherical wave functions, whose coefficients follow from the boundary
conditions at the sphere surface and the radiation condition at infinity
(Mishchenko and Travis, 2008). Its assumptions are:

* a homogeneous, isotropic sphere;
* a plane incident wave (a laser beam much wider than the particle);
* a homogeneous, non-absorbing host medium;
* observation in the far field;
* for a suspension, independent scattering by the particles, i.e. a medium
  "optically infinitely dilute" (Mie, 1908, §22).

Size parameter and refractive index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solution depends on two dimensionless quantities: the size parameter
:math:`x`, ratio of the sphere circumference to the wavelength in the medium
(Mie's :math:`\alpha = 2\pi\rho/\lambda'`), and the refractive index :math:`m`
of the particle relative to the medium:

.. math::
   :label: eq-x-m

   x = \frac{2\pi r}{\lambda_{med}} = \frac{\pi d\, n_{med}}{\lambda}, \qquad
   m = \frac{n_p + i k_p}{n_{med}}

where :math:`r` and :math:`d` are the particle radius and diameter, and
:math:`\lambda_{med} = \lambda/n_{med}` is the wavelength in the medium. For the
polystyrene beads, :math:`n_p = 1.60` at 515 nm (no absorption) and the
refractive index of water is computed with the dispersion formula used in the
notebooks (:math:`\lambda` in nm):

.. math::
   :label: eq-nw

   n_w(\lambda) = 1.3199 + \frac{6878}{\lambda^2} - \frac{1.132\times10^{9}}{\lambda^4}
                  + \frac{1.11\times10^{14}}{\lambda^6}

which gives :math:`n_w \approx 1.336` and :math:`m \approx 1.198` at 515 nm.

Mie coefficients
~~~~~~~~~~~~~~~~

The scattered field is the sum of partial waves of order :math:`n`, weighted
by the Mie coefficients :math:`a_n` (electric) and :math:`b_n` (magnetic).
In the computationally efficient form of Bohren and Huffman (1983, Eqs.
4.88-4.89), as implemented in PyMieScatt (Sumlin et al., 2018):

.. math::
   :label: eq-an-bn

   a_n &= \frac{\left[\dfrac{D_n(mx)}{m} + \dfrac{n}{x}\right]\psi_n(x) - \psi_{n-1}(x)}
              {\left[\dfrac{D_n(mx)}{m} + \dfrac{n}{x}\right]\xi_n(x) - \xi_{n-1}(x)} \\[1ex]
   b_n &= \frac{\left[m D_n(mx) + \dfrac{n}{x}\right]\psi_n(x) - \psi_{n-1}(x)}
              {\left[m D_n(mx) + \dfrac{n}{x}\right]\xi_n(x) - \xi_{n-1}(x)}

where :math:`\psi_n(\rho) = \rho\, j_n(\rho)` and
:math:`\xi_n(\rho) = \rho\, h_n^{(1)}(\rho)` are Riccati-Bessel functions
(:math:`j_n` and :math:`h_n^{(1)}` being the spherical Bessel and Hankel
functions), and :math:`D_n(\rho) = \psi_n'(\rho)/\psi_n(\rho)` is their
logarithmic derivative, computed by downward recurrence. The series converge
after about

.. math::
   :label: eq-nmax

   n_{max} = x + 4x^{1/3} + 2

terms (Wiscombe, 1980), i.e. about 120 terms for :math:`x = 100`, the upper
bound of the computations used for the beads. Narrow resonances in :math:`a_n` and :math:`b_n` as a
function of :math:`x` cause the fine "ripple" structure of monodisperse Mie
results, which vanishes when integrating over a size distribution
(Mishchenko and Travis, 2008).

Amplitude functions and scattering matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The far-field amplitudes of the scattered electric field perpendicular
(:math:`S_1`) and parallel (:math:`S_2`) to the scattering plane are

.. math::
   :label: eq-s1-s2

   S_1(\theta) &= \sum_{n=1}^{n_{max}} \frac{2n+1}{n(n+1)}\,(a_n \pi_n + b_n \tau_n) \\
   S_2(\theta) &= \sum_{n=1}^{n_{max}} \frac{2n+1}{n(n+1)}\,(a_n \tau_n + b_n \pi_n)

The angular functions :math:`\pi_n = P_n^1(\cos\theta)/\sin\theta` and
:math:`\tau_n = dP_n^1(\cos\theta)/d\theta`, with :math:`P_n^1` the associated
Legendre functions, follow from the upward recurrences
(:math:`\mu = \cos\theta`, :math:`\pi_0 = 0`, :math:`\pi_1 = 1`):

.. math::
   :label: eq-pi-tau

   \pi_n = \frac{2n-1}{n-1}\,\mu\,\pi_{n-1} - \frac{n}{n-1}\,\pi_{n-2}, \qquad
   \tau_n = n\mu\,\pi_n - (n+1)\,\pi_{n-1}

For a sphere, the scattering matrix has the form of :eq:`eq-mueller` with
:math:`S_{22} = S_{11}` and :math:`S_{44} = S_{33}`, and its four independent
elements are

.. math::
   :label: eq-sij

   S_{11} &= \tfrac{1}{2}\left(|S_2|^2 + |S_1|^2\right), \qquad
   S_{12} = \tfrac{1}{2}\left(|S_2|^2 - |S_1|^2\right), \\
   S_{33} &= \tfrac{1}{2}\left(S_2^* S_1 + S_2 S_1^*\right), \qquad
   S_{34} = \tfrac{i}{2}\left(S_1 S_2^* - S_2 S_1^*\right)

Mie (1908, Eq. 78) gave the intensities of the two polarized components as
:math:`\lambda'^2/(4\pi^2 r^2)\,|\dots|^2`, i.e. :math:`|S_{1,2}|^2/(k^2 r^2)`
with the notation of :eq:`eq-stokes`.

Efficiencies
~~~~~~~~~~~~

The extinction and scattering cross sections, normalized by the geometrical
cross section :math:`\pi r^2`, and the asymmetry parameter of a single sphere
are (Sumlin et al., 2018, Eqs. S2):

.. math::
   :label: eq-q

   Q_{ext} &= \frac{2}{x^2} \sum_{n=1}^{n_{max}} (2n+1)\,\mathrm{Re}\{a_n + b_n\}, \\
   Q_{sca} &= \frac{2}{x^2} \sum_{n=1}^{n_{max}} (2n+1)\left(|a_n|^2 + |b_n|^2\right),
   \qquad Q_{abs} = Q_{ext} - Q_{sca}, \\
   g &= \frac{4}{Q_{sca}\, x^2} \left[\sum_{n=1}^{n_{max}} \frac{n(n+2)}{n+1}
        \mathrm{Re}\{a_n a_{n+1}^* + b_n b_{n+1}^*\} \right. \\
     &\qquad \left. + \sum_{n=1}^{n_{max}} \frac{2n+1}{n(n+1)} \mathrm{Re}\{a_n b_n^*\}\right]

They are consistent with the scattering matrix through
:math:`\pi r^2 Q_{sca} = k^{-2}\int_{4\pi} S_{11}\, d\Omega`.

.. note::

   The backscattering efficiency
   :math:`Q_{back} = x^{-2}\left|\sum_n (2n+1)(-1)^n (a_n - b_n)\right|^2`
   returned by Mie codes refers to the exact backward direction
   (:math:`\theta = 180^\circ`, "radar" backscatter). It differs from the
   hemispherical backscattering coefficient :math:`b_b` of :eq:`eq-b`,
   which must be integrated from the phase function.

Rayleigh limit
~~~~~~~~~~~~~~

For particles much smaller than the wavelength (:math:`x \ll 1`), only the
first terms of the series matter and

.. math::
   :label: eq-rayleigh

   Q_{sca} &= \frac{8x^4}{3}\left|\frac{m^2-1}{m^2+2}\right|^2 \\
   S_{11} &\propto 1 + \cos^2\theta, \qquad
   \frac{S_{12}}{S_{11}} = -\frac{\sin^2\theta}{1+\cos^2\theta}

so that :math:`\tilde{b}_b = 0.5`, :math:`g = 0` and the light scattered at
90 degrees is fully polarized. These values are a useful sanity check for the
smallest beads.

Polydispersions
---------------

For a suspension of spheres with a number size distribution :math:`n(D)`
(m\ :sup:`-4`, number of particles per unit volume and per unit diameter),
the elements of the volume scattering matrix and the scattering coefficient
are obtained by integrating the single-particle quantities:

.. math::
   :label: eq-psd

   P_{ij}(\theta) &= \frac{1}{k^2} \int_0^{\infty} S_{ij}(\theta; m, x(D))\, n(D)\, dD \\
   b &= \int_0^{\infty} \frac{\pi D^2}{4}\, Q_{sca}(m, x(D))\, n(D)\, dD

In the bead notebooks, the Mie scattering matrix is computed once for size
parameters :math:`x` between 1 and 100 and cached; for each bead sample it is
integrated over a narrow log-normal distribution centered on the nominal bead
radius (coefficient of variation below 1 %). The normalized elements
:math:`S_{12}/S_{11}`, :math:`S_{33}/S_{11}` and :math:`S_{34}/S_{11}` are
compared with ``p12`` and ``p22`` (equal to 1 for spheres), and the integrated
:math:`S_{11}` is converted to a phase function with :eq:`eq-pf-norm`:

.. math::
   :label: eq-pf-mie

   \tilde{\beta}_{Mie}(\theta) = \frac{\bar{S}_{11}(\theta)}
   {2\pi \int_0^{\pi} \bar{S}_{11}(\theta) \sin\theta \, d\theta}

The LISST-VSF phase function is normalized in the same way over its valid
angular range, and the backscattering ratios of :eq:`eq-bb-ratio` are
compared, with the Mie integrals computed over either the full or the
LISST-VSF angular range.

References
----------

* Bohren, C. F., and D. R. Huffman, 1983: *Absorption and Scattering of Light
  by Small Particles*. John Wiley & Sons, 530 pp.
* Mie, G., 1908: Beiträge zur Optik trüber Medien, speziell kolloidaler
  Metallösungen. *Annalen der Physik*, **330** (25), 377-445,
  `doi:10.1002/andp.19083300302 <https://doi.org/10.1002/andp.19083300302>`_.
* Mishchenko, M. I., and L. D. Travis, 2008: Gustav Mie and the evolving
  discipline of electromagnetic scattering by particles. *Bulletin of the
  American Meteorological Society*, **89**, 1853-1861,
  `doi:10.1175/2008BAMS2632.1 <https://doi.org/10.1175/2008BAMS2632.1>`_.
* Sumlin, B. J., W. R. Heinson, and R. K. Chakrabarty, 2018: Retrieving the
  aerosol complex refractive index using PyMieScatt: A Mie computational
  package with visualization capabilities. *Journal of Quantitative
  Spectroscopy and Radiative Transfer*, **205**, 127-134,
  `doi:10.1016/j.jqsrt.2017.10.012 <https://doi.org/10.1016/j.jqsrt.2017.10.012>`_.
* Wiscombe, W. J., 1980: Improved Mie scattering algorithms. *Applied Optics*,
  **19**, 1505-1509, `doi:10.1364/AO.19.001505 <https://doi.org/10.1364/AO.19.001505>`_.
