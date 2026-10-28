# Changelog

All notable changes to PyLisst are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Basics tutorials: `quickstart_lisst_vsf` (reading, processing, quality control,
  VSF and Mueller matrix plots, phase function, backscattering ratio, netCDF export)
  and `quickstart_lisst_200x` (LISST-200X reader); `beads_process_basic` added to the
  tutorials.
- Workflow diagram and per-notebook descriptions on the Advanced tutorials page.
- This changelog.

### Changed
- README and installation page rewritten with best-practice installation:
  isolated `venv`/conda environment, install from GitHub with optional version
  pinning, editable development install, installation check, optional
  dependencies (`mie_perso`, Jupyter), upgrade/uninstall instructions.

### Fixed
- Ring data of the second laser polarization: `driver.reader` built `rings2`
  from `rings1`, and `process.process_forward_angles` used `rings1` for both
  polarizations. The ring VSF now averages both ring channels, as in the
  Sequoia MATLAB code (changes the absolute VSF and hence `P11`).
- Eyeball attenuation path: the distance along the beam to the sample volume
  is `R cot(theta)`, not `R arctan(theta)` (Sequoia `lisstvsf_makep.m`). This
  increases the attenuation correction at large angles, hence `P11` and
  `b_b/b` for turbid samples.
- Half-wave plate correction applied twice to the background laser reference
  of the second polarization (in `get_attenuation` and again in
  `process_large_angles`), which biased `beam_c` of the second polarization
  (e.g. 7.06 instead of 4.49 m-1) and the eyeball signals derived from it.
  The correction is now applied once, in the new
  `process.correct_half_wave_plate`, on copies of the input data: the
  `driver` objects are no longer modified in place, so processing the same
  data twice gives the same result.
- `get_matrix_terms`: P22 written as the mean of the two estimates, each
  normalized as in the Sequoia code (results unchanged).
- Lower bound of the backscattering integral (pi/2 instead of 2 pi) in the
  `post_processing` notebook.

### Removed
- `basics_quickstart` tutorial page, replaced by the quickstart notebooks.

## [1.0.2] - 2026-10-08

### Added
- Reader and processor of LISST-200X `.RBN` files (`pylisst.lisst_x`).
- Sphinx documentation (Read the Docs): installation, processing chain,
  API reference and tutorials.
- Packaging with `pyproject.toml` (replaces `setup.py`).

### Changed
- Cleaning and optimization of the processing chain and calibration;
  updated beads, HABLAB and post-processing notebooks.

## [1.0.1] - 2023-10-02

### Changed
- Background Z measurement removed before the attenuation correction.

## [1.0.0] - 2022-05-02

### Added
- First complete version: LISST-VSF binary reader, calibration and processing
  of rings and large (eyeball) angles into VSF and Mueller matrix terms
  (P11, P12, P22); Mie theory checks with polystyrene beads.

[Unreleased]: https://github.com/Tristanovsk/pylisst/compare/8ca4e31...HEAD
[1.0.2]: https://github.com/Tristanovsk/pylisst/tree/8ca4e31
[1.0.1]: https://github.com/Tristanovsk/pylisst/tree/cfe2595
[1.0.0]: https://github.com/Tristanovsk/pylisst/tree/c03bc59
