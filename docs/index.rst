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

DJINN: Deep Jointly-Informed Neural Networks's documentation!
=============================================================
Building djinn regression models with
PyTorch and sklearn.

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
``Pipeline``, ``cross_val_score`` and ``GridSearchCV``.

For more info, see `the paper <https://arxiv.org/abs/1707.00784>`_.

Package Contents
================

.. toctree::
   install
   djinn
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


License & Usage
================

.. toctree::
   djinn_license
   :maxdepth: 1
