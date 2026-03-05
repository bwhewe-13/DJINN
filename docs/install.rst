Installation Guide
==================
`djinn` requires scikit-learn and PyTorch.

Installing with pip
-------------------
You can install djinn via pip.

1. clone the repo from: github.com/LLNL/djinn ::

   $ cd djinn_parent_dir
   $ git clone https://github.com/bwhewe-13/DJINN.git

2. pip user install (eg)::

   $ cd DJINN
   $ pip install --user .

3. test out loading djinn::

   $ cd
   $ python -c "from djinn import DJINN_Regressor"

4. try out an example::

   $ python DJINN/examples/djinn_regression_example.py
