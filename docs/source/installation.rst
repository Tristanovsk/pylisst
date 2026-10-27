Installation
============

Requirements
------------

- Python >= 3.9
- ``numpy``, ``scipy``, ``pandas``, ``xarray``, ``lmfit`` and ``matplotlib``
  (installed automatically by ``pip``, see ``pyproject.toml``)

Create an environment
---------------------

Always install PyLisst in a dedicated, isolated environment rather than in
your system Python.

With the standard library ``venv``:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip

or with conda / mamba:

.. code-block:: bash

   conda create -n pylisst python=3.11
   conda activate pylisst

Install the package
-------------------

**User install**: directly from GitHub, no clone needed:

.. code-block:: bash

   python -m pip install "git+https://github.com/Tristanovsk/pylisst.git"

To pin a specific release (recommended for reproducible processing):

.. code-block:: bash

   python -m pip install "git+https://github.com/Tristanovsk/pylisst.git@<tag-or-commit>"

**Development install**: clone `the repository
<https://github.com/Tristanovsk/pylisst>`_ and install in editable mode, so
that changes to the source are picked up without reinstalling:

.. code-block:: bash

   git clone https://github.com/Tristanovsk/pylisst.git
   cd pylisst
   python -m pip install -e .

Check the installation
----------------------

.. code-block:: bash

   python -c "import pylisst; print(pylisst.__version__)"

Optional dependencies
---------------------

The particle size distribution inversion of LISST-200X data
(:mod:`pylisst.lisst_x`) relies on the external ``mie_perso`` package, which
is not installed automatically. Install it separately in the same environment
if you need that feature.

To run the tutorial notebooks, install Jupyter:

.. code-block:: bash

   python -m pip install jupyterlab

Upgrade / uninstall
-------------------

.. code-block:: bash

   python -m pip install --upgrade "git+https://github.com/Tristanovsk/pylisst.git"
   python -m pip uninstall pylisst

Building the documentation
--------------------------

.. code-block:: bash

   python -m pip install -e .
   python -m pip install -r docs/requirements.txt
   cd docs
   make html

The HTML pages are written in ``docs/build/html``.
