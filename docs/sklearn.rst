Using DJINN with scikit-learn
=============================
:class:`~djinn.DJINN_Regressor` and :class:`~djinn.DJINN_Classifier` are
scikit-learn estimators, so they work anywhere a scikit-learn model does.
:doc:`examples/05_scikit_learn` runs everything on this page.

Settings and ``clone``
----------------------
Every setting is a constructor argument, so :meth:`get_params`,
:meth:`set_params` and :func:`sklearn.base.clone` work as usual. ``clone``
copies the settings, not the trained networks.

.. code-block:: python

    from sklearn.base import clone
    from djinn import DJINN_Regressor

    model = DJINN_Regressor(n_trees=3, max_tree_depth=6, random_state=0)
    bigger = clone(model).set_params(n_trees=5)

Inputs
------
``X`` can be a NumPy array, a list or a pandas DataFrame. Column names from a
DataFrame are kept in ``feature_names_in_``, and scikit-learn warns if you
predict on data with different columns.

Pipelines
---------
DJINN scales each input to [0, 1] using the training minimum and maximum, so it
doesn't need standardized inputs. It does need inputs without extreme
outliers. A single huge value squashes the rest of that feature into a small
range, and a test point far outside the training range gets extrapolated in a
straight line by the ReLU networks, which can give wild predictions. A
:class:`~sklearn.preprocessing.QuantileTransformer` in front of DJINN fixes
both. Text or categorical columns also need encoding:

.. code-block:: python

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

    prep = ColumnTransformer(
        [
            ("region", OneHotEncoder(), ["region"]),
            ("numeric", QuantileTransformer(n_quantiles=500), numeric_columns),
        ]
    )
    pipe = Pipeline([("prep", prep), ("djinn", DJINN_Regressor(n_trees=3))])
    pipe.fit(X_train, y_train)

On the California housing data in :doc:`examples/05_scikit_learn`, one row
with an average household size over 1,000 drops DJINN's R² on its
cross-validation fold below zero. With the quantile transform, DJINN beats a
random forest on every fold.

Cross-validation and grid search
--------------------------------
:func:`~sklearn.model_selection.cross_val_score` and
:class:`~sklearn.model_selection.GridSearchCV` fit a fresh clone for every
fold. If ``learning_rate`` is ``None``, each of those fits runs the
hyperparameter search first, which multiplies the cost. Run
:meth:`~djinn.DJINN_Regressor.get_hyperparameters` once on the training data
and set ``learning_rate`` (and ``epochs``) in the constructor:

.. code-block:: python

    from sklearn.model_selection import GridSearchCV

    model = DJINN_Regressor(learning_rate=0.005, epochs=300, random_state=0)
    search = GridSearchCV(
        model, {"max_tree_depth": [4, 6, 8], "n_trees": [1, 3, 5]}, cv=5
    )
    search.fit(X_train, y_train)

Inside a pipeline, prefix the parameter with the step name, as in
``djinn__max_tree_depth``.

``score()`` is R² for :class:`~djinn.DJINN_Regressor` and accuracy for
:class:`~djinn.DJINN_Classifier`, which is what ``cross_val_score`` and
``GridSearchCV`` use unless you pass ``scoring``.

Saving
------
Fitted models and pipelines can be saved with :mod:`pickle` or
:mod:`joblib`. As with any pickle, load them with the same versions of DJINN,
scikit-learn and PyTorch.

:meth:`~djinn.DJINN_Regressor.save` and :func:`djinn.load` save only the DJINN
model, as PyTorch checkpoints and a JSON file. Use them to keep a model long
term or to keep training it later with
:meth:`~djinn.DJINN_Regressor.continue_training`.

Estimator checks
----------------
The test suite runs scikit-learn's
:func:`~sklearn.utils.estimator_checks.parametrize_with_checks` on both
models. A few checks are expected to fail:

- ``check_fit2d_1feature``, ``check_estimators_pickle``,
  ``check_pipeline_consistency``, and ``check_classifiers_classes`` for the
  classifier, fit on a handful of samples or on data that one split separates.
  The trees only grow to depth 1, which is too shallow to build a network from.
  Pickling and pipelines are tested separately on realistic data.
- ``check_methods_subset_invariance`` compares predictions made in batches
  with predictions made one row at a time. The networks run in float32, so
  these can differ by about 1e-7, which is more than the check allows.
