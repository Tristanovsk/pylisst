Installation
============

Clone `the repository <https://github.com/Tristanovsk/pylisst>`_ and install
the package from the local copy:

.. code-block:: bash

   git clone https://github.com/Tristanovsk/pylisst.git
   cd pylisst
   python3 -m pip install .

Use ``python3 -m pip install -e .`` for a development (editable) install.

Dependencies
------------

``numpy``, ``scipy``, ``pandas``, ``xarray``, ``lmfit`` and ``matplotlib``
are installed automatically (see ``pyproject.toml``).

Building the documentation
--------------------------

.. code-block:: bash

   python3 -m pip install -e .
   python3 -m pip install -r docs/requirements.txt
   cd docs
   make html

The HTML pages are written in ``docs/build/html``.
