.. #
.. # Copyright (c) 2018, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # Written by K. Humbird (humbird1@llnl.gov), L. Peterson (peterson76@llnl.gov).
.. #
.. # LLNL-CODE-754815
.. #
.. # All rights reserved.
.. #
.. # This file is part of DJINN.
.. #
.. # For details, see github.com/LLNL/djinn.
.. #
.. # For details about use and distribution, please read DJINN/LICENSE .
.. #
.. Deep Jointly-Informed Neural Networks documentation master file, created by
   sphinx-quickstart on Tue Dec 19 11:24:39 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DJINN: Deep Jointly-Informed Neural Networks
============================================
DJINN builds an ensemble of neural networks from a random forest. Each
decision tree sets the architecture and the starting weights of one network,
and then the networks are trained with PyTorch. Most of the effort of
designing a network is done by the trees, so the only settings to tune are the
number of trees and their depth. With dropout, the ensemble also gives
uncertainty estimates.

DJINN was introduced by Humbird, Peterson and McClarren in
`Deep Neural Network Initialization With Decision Trees
<https://doi.org/10.1109/TNNLS.2018.2869694>`_ (`arXiv
<https://arxiv.org/abs/1707.00784>`_). For an application, see
`Whewell and McClarren (2022) <https://doi.org/10.1016/j.anucene.2022.109276>`_,
which used DJINN to reduce the data needed for neutron transport calculations.

Example
-------
.. code-block:: python

    from djinn import DJINN_Regressor, load

    # Fit a regression model and predict something new
    model = DJINN_Regressor(n_trees=3, random_state=0)
    model.fit(X, y)
    y_new = model.predict(x_new)

    # Save it, then reload and keep training on more data
    model.save("my_model")
    model = load("my_model")
    model.continue_training(X1, y1, training_epochs=100,
                            learning_rate=0.005, batch_size=32)

Both models are scikit-learn estimators, so they can be used in
``Pipeline``, ``cross_val_score`` and ``GridSearchCV``. Install with
``pip install djinnml``.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   install
   examples/01_quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   theory
   usage
   sklearn

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: About

   citing
   changelog
   djinn_license

Indices
-------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
