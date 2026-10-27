# PyLisst
Scientific code to process and visualize LISST-VSF data.

PyLisst reads raw binary files of the Sequoia LISST-VSF instrument, applies
background (clean-water) and attenuation corrections and computes the volume
scattering function (VSF) together with the normalized Mueller matrix terms
(P11, P12, P22). A reader for LISST-200X `.RBN` files is also provided.

Full documentation: https://pylisst.readthedocs.io

## Requirements

- Python >= 3.9
- `numpy`, `scipy`, `pandas`, `xarray`, `lmfit`, `matplotlib`
  (installed automatically by `pip`, see `pyproject.toml`)

## Installation

Always install PyLisst in a dedicated, isolated environment rather than in
your system Python.

### 1. Create an environment

With the standard library `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

or with conda / mamba:

```bash
conda create -n pylisst python=3.11
conda activate pylisst
```

### 2. Install the package

**User install** — directly from GitHub, no clone needed:

```bash
python -m pip install "git+https://github.com/Tristanovsk/pylisst.git"
```

To pin a specific release (recommended for reproducible processing):

```bash
python -m pip install "git+https://github.com/Tristanovsk/pylisst.git@<tag-or-commit>"
```

**Development install** — clone the repository and install in editable mode,
so that changes to the source are picked up without reinstalling:

```bash
git clone https://github.com/Tristanovsk/pylisst.git
cd pylisst
python -m pip install -e .
```

### 3. Check the installation

```bash
python -c "import pylisst; print(pylisst.__version__)"
```

### Optional dependencies

The particle size distribution inversion of LISST-200X data
(`pylisst.lisst_x`) relies on the external `mie_perso` package, which is not
installed automatically. Install it separately in the same environment if
you need that feature.

To run the tutorial notebooks, install Jupyter:

```bash
python -m pip install jupyterlab
```

### Upgrade / uninstall

```bash
python -m pip install --upgrade "git+https://github.com/Tristanovsk/pylisst.git"
python -m pip uninstall pylisst
```

## Documentation

The documentation is hosted at https://pylisst.readthedocs.io.
To build it locally (from a development install):

```bash
python -m pip install -e .
python -m pip install -r docs/requirements.txt
cd docs
make html
```

The HTML pages are written in `docs/build/html`.

## License

MIT — see [LICENSE](https://github.com/Tristanovsk/pylisst/blob/master/LICENSE).
