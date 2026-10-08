# PyLisst
Scientific code to process and visualize LISST-VSF data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

Clone [the repository](https://github.com/Tristanovsk/pylisst) and install the package from the local copy:

```
git clone https://github.com/Tristanovsk/pylisst.git
cd pylisst
python3 -m pip install .
```

For a development (editable) install:

```
python3 -m pip install -e .
```

### Documentation

Build the documentation locally with:

```
python3 -m pip install -e .
python3 -m pip install -r docs/requirements.txt
cd docs
make html
```
