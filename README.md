
Deep Jointly-Informed Neural Networks
======================================
**DJINN: Deep jointly-informed neural networks**


[![Tests](https://github.com/bwhewe-13/DJINN/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/bwhewe-13/DJINN/actions/workflows/tests.yml)
[![Formatting](https://github.com/bwhewe-13/DJINN/actions/workflows/formatting.yml/badge.svg?branch=master)](https://github.com/bwhewe-13/DJINN/actions/workflows/formatting.yml)
[![codecov](https://codecov.io/gh/bwhewe-13/DJINN/branch/master/graph/badge.svg)](https://codecov.io/gh/bwhewe-13/DJINN)
[![Docs](https://github.com/bwhewe-13/DJINN/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/bwhewe-13/DJINN/actions/workflows/docs.yml)
[![License: BSD](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/bwhewe-13/DJINN/blob/master/LICENSE)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)


Fork notice: This repository is a fork and continuation of LLNL's DJINN project (Deep Jointly-Informed Neural Networks) originally developed by Kelli D. Humbird (humbird1@llnl.gov). The original project is available at https://github.com/LLNL/djinn and is distributed under the license found in `LICENSE`. This fork is maintained by Ben Whewell (ben.whewell@pm.me) — https://github.com/bwhewe-13/DJINN


DJINN is an easy-to-use algorithm for training deep neural networks on supervised regression tasks.
For additional information, refer to the paper "Deep neural network initialization with decision trees", cited below.



Getting Started
-----------
Original DJINN required TensorFlow. This fork is implemented with PyTorch.

Requirements:

- Python 3.8+
- PyTorch
- scikit-learn

Install from source:

```bash
git clone https://github.com/bwhewe-13/DJINN.git
cd DJINN
python -m pip install --upgrade pip
python -m pip install .
```

Try it out using the examples in [examples](./examples):

- `python examples/djinn_regression_example.py`
- `python examples/djinn_classification_example.py`
- `python examples/djinn_multiout_example.py`

Notes:

- The scikit-learn version used when training a DJINN model should match the
    version used when loading/evaluating that saved model.
- Some example workflows may require `matplotlib`:

    ```bash
    python -m pip install matplotlib
    ```


Development
-----------
Set up a local development environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Run quality checks and tests:

```bash
black --check djinn tests examples
isort --check-only djinn tests examples
flake8
pytest
```

Enable pre-commit hooks (optional, recommended):

```bash
pre-commit install
pre-commit run --all-files
```

Build docs locally:

```bash
python -m pip install sphinx
cd docs
make html
```



### Documentation
To view the DJINN documentation:

```
cd docs
make html
```
Open docs/_build/html/index.html in a browser


Source Repo
-----------

DJINN is available at https://github.com/LLNL/DJINN


Citing DJINN
-----------
If you use DJINN in your research, please cite the following paper:

K. D. Humbird, J. L. Peterson and R. G. Mcclarren, "Deep Neural Network Initialization With Decision Trees," in IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 5, pp. 1286-1295, May 2019.
doi: 10.1109/TNNLS.2018.2869694,
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8478232&isnumber=8695188




Release
-----------
Copyright (c) 2018, Lawrence Livermore National Security, LLC.

Produced at the Lawrence Livermore National Laboratory

Written by K. Humbird (humbird1@llnl.gov), L. Peterson (peterson76@llnl.gov).

LLNL-CODE-754815   OCEC-18-117

All rights reserved.

Unlimited Open Source- BSD Distribution.

For release details and restrictions, please read the RELEASE, LICENSE, and NOTICE files, linked below:
- [RELEASE](./RELEASE)
- [LICENSE](./LICENSE)
- [NOTICE](./NOTICE)
