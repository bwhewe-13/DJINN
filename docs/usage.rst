User guide
==========
This page covers the main ways to use DJINN. :doc:`theory` explains what each
step does, and :doc:`api` lists every method.

Choosing a model
----------------
:class:`~djinn.DJINN_Regressor` predicts one or more continuous targets.
:class:`~djinn.DJINN_Classifier` predicts class labels, which can be any
values (numbers, strings, ``1..k``). Both take the same settings:

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - Meaning
   * - ``n_trees``
     - 1
     - Number of trees in the random forest, and so the number of networks in
       the ensemble.
   * - ``max_tree_depth``
     - 4
     - Maximum depth of each tree. A network has ``max_tree_depth - 1``
       hidden layers.
   * - ``dropout_keep_prob``
     - 1.0
     - Chance of keeping each hidden neuron. Below 1 turns on dropout, which
       is needed for :ref:`uncertainty <usage-uncertainty>`.
   * - ``learning_rate``
     - ``None``
     - ``None`` runs the hyperparameter search in :meth:`fit`.
   * - ``epochs``
     - ``None``
     - Training epochs. Chosen by the search when ``learning_rate`` is
       ``None``, otherwise 1000.
   * - ``batch_size``
     - ``None``
     - Minibatch size. ``None`` uses 5% of the samples.
   * - ``weight_decay``
     - 1e-8
     - L2 penalty on the weights.
   * - ``random_state``
     - ``None``
     - Seed for the forest, the starting weights and training.
   * - ``device``
     - ``"cpu"``
     - Torch device used for training and prediction.
   * - ``verbose``
     - 0
     - Print progress when greater than 0.

``n_trees`` and ``max_tree_depth`` are the two settings worth tuning. Deeper
trees give bigger networks that can fit more complex data but need more data
to train.

Preparing the data
------------------
Inputs, and targets for regression, are scaled to [0, 1] with the training
minimum and maximum, so there's no need to standardize them first. Two things
are worth doing:

- **Remove extreme outliers.** One very large value squashes the rest of a
  feature into a small range, and predictions for inputs far outside the
  training range can be wildly off, as with any ReLU network. A
  :class:`~sklearn.preprocessing.QuantileTransformer`, a log transform or
  clipping fixes this. See :doc:`sklearn`.
- **Encode categorical features**, for example with
  :class:`~sklearn.preprocessing.OneHotEncoder`.

Training
--------
:meth:`~djinn.DJINN_Regressor.fit` does everything in one call:

.. code-block:: python

    from djinn import DJINN_Regressor

    model = DJINN_Regressor(n_trees=3, max_tree_depth=6, random_state=0)
    model.fit(X_train, y_train)

With ``learning_rate=None``, ``fit()`` first searches for a learning rate,
number of epochs and batch size. The search trains the network from the first
tree about twenty times, so it usually takes longer than the training itself.
When you'll fit the same data more than once, run the search once with
:meth:`~djinn.DJINN_Regressor.get_hyperparameters` and pass the results to the
constructor:

.. code-block:: python

    hyper = DJINN_Regressor(max_tree_depth=6).get_hyperparameters(
        X_train, y_train, seed=0
    )
    model = DJINN_Regressor(
        n_trees=3,
        max_tree_depth=6,
        learning_rate=hyper["learning_rate"],
        epochs=hyper["epochs"],
        batch_size=hyper["batch_size"],
        random_state=0,
    )
    model.fit(X_train, y_train)

:meth:`~djinn.DJINN_Regressor.train` is the lower-level method behind ``fit()``.
It takes the training options as arguments instead of from the constructor.

.. note::

    Passing training options to ``fit()``, as in
    ``fit(X, y, epochs=100)``, still works but is deprecated and will be
    removed in 2.0. Set them in the constructor instead.

Predicting
----------
:meth:`~djinn.DJINN_Regressor.predict` runs every network once with dropout off
and averages them. It returns shape ``(n_samples,)`` when the model was trained
on a 1-D ``y``, and ``(n_samples, n_outputs)`` otherwise.

For classifiers, ``predict()`` returns labels in the form they were given, and
:meth:`~djinn.DJINN_Classifier.predict_proba` returns one column per class, in
the order of ``classes_``.

``score()`` is R² for the regressor and accuracy for the classifier, the same
as other scikit-learn estimators.

.. _usage-uncertainty:

Uncertainty
-----------
Train with dropout (``dropout_keep_prob`` around 0.95) and several trees, then
call :meth:`~djinn.DJINN_Regressor.bma_predict`. It runs ``n_iters`` passes
through each network with dropout on:

.. code-block:: python

    model = DJINN_Regressor(n_trees=5, dropout_keep_prob=0.95, random_state=0)
    model.fit(X_train, y_train)

    result = model.bma_predict(X_test, n_iters=50, seed=0)
    result["lower"], result["middle"], result["upper"]  # 25th, 50th, 75th percentiles
    result["predictions"]  # all samples, (n_iters * n_trees, n_test, n_outputs)

:meth:`~djinn.DJINN_Regressor.bayesian_predict` gives the same percentiles as a
tuple, with the samples grouped by network. :doc:`examples/04_uncertainty`
shows how to check whether the intervals can be trusted.

Saving and loading
------------------
:meth:`~djinn.DJINN_Regressor.save` writes one PyTorch checkpoint per network
to a folder, and the settings and scalers to a ``.json`` file beside it.
:func:`djinn.load` reads them back, as a regressor or classifier to match what
was saved:

.. code-block:: python

    from djinn import load

    model.save("my_model")  # writes my_model/ and my_model.json
    model = load("my_model")

A loaded model can keep training on new or the same data with
:meth:`~djinn.DJINN_Regressor.continue_training`. The checkpoints are updated
in place:

.. code-block:: python

    model.continue_training(
        X_train, y_train, training_epochs=100, learning_rate=0.005, batch_size=32
    )

Fitted models can also be pickled. That's the easiest way to save a model
inside a scikit-learn pipeline. See :doc:`sklearn`.

Reproducibility
---------------
With ``random_state`` set, the forest, the starting weights, the minibatches
and dropout are all seeded, so fitting the same data twice on the same machine
gives the same model. Results can still differ slightly between machines and
library versions. ``predict()`` doesn't use any randomness. Pass ``seed`` to
``bma_predict()`` and ``bayesian_predict()`` to make the sampling repeatable.
