# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `DJINN_Regressor` and `DJINN_Classifier` are scikit-learn estimators. They
  work with `clone`, `Pipeline`, `cross_val_score` and `GridSearchCV`, and
  pass scikit-learn's estimator checks except a few that use data too small
  to grow a tree deeper than 1.
- `epochs`, `learning_rate`, `batch_size`, `weight_decay`, `random_state`,
  `device` and `verbose` constructor options.
- `DJINN_Classifier.predict_proba()` and `classes_`. Class labels can be any
  values, such as strings or 1..k.
- Lists and pandas DataFrames are accepted as input, and DataFrame column
  names are stored in `feature_names_in_`.

### Changed

- `fit(X, y)` takes its options from the constructor and returns the model.
  Passing options to `fit()` still works but is deprecated and will be
  removed in 2.0.
- `fit()` and `train()` no longer write files by default. Use `save()`, or
  pass `save_model=True` to `train()`.
- `predict()` returns shape `(n,)` when the model was trained on a 1-D
  target, and regression predictions are float64.
- `predict()` rejects 1-D input; pass one sample as `X[[i]]`.
- Progress messages are off unless `verbose=1`. Hitting the epoch limit in
  the hyperparameter search raises a `ConvergenceWarning` instead of
  printing.
- `get_hyperparameters()` runs on the model's `device`; it used to pick the
  GPU whenever one was available.
- `DJINN_Classifier` no longer subclasses `DJINN_Regressor`, and the
  constructors no longer accept `**kwargs`.
- Requires scikit-learn 1.6 or newer.

### Fixed

- `load()` returns a `DJINN_Classifier` for saved classifiers. They used to
  come back as regressors and fail to predict.
- Fitting again on new data rescales to that data instead of reusing the
  scaling from the first fit.

## [1.1.2] - 2026-10-05

### Fixed

- `get_hyperparameters(seed=...)` is now reproducible. The learning-rate
  search drew its initial biases from OS entropy and ran dropout on the
  unseeded global torch RNG, so the same seed could select a different
  learning rate on each call.

### Changed

- Minibatches are now drawn the way the original TensorFlow DJINN draws
  them: each epoch samples `len(X) // batch_size` full batches with
  replacement, instead of a shuffled pass over every row. This closes a
  ~0.02 R² gap to the TensorFlow baseline on the diabetes benchmark. Trained
  models and selected hyperparameters will differ from 1.1.1 for the same
  seed.
- CI runs on Ubuntu 26.04 with Node 24 versions of the GitHub Actions.

## [1.1.1] - 2026-10-04

### Fixed

- `save()` writes the trained models held in memory instead of copying
  `<model_path>/<model_name>` from disk. It previously failed when `train()`
  ran with `save_model=False` and could save a stale model from an earlier
  run.
- `save()` raises `RuntimeError` when no model has been trained or loaded.

## [1.1.0] - 2026-10-04

First release on PyPI as `djinnml`. DJINN is reimplemented in PyTorch; the
public API (`DJINN_Regressor`, `DJINN_Classifier`, `train`, `fit`, `predict`,
`bma_predict`, `save`, `load`) matches the original TensorFlow version.

### Added

- Statistical comparison suite against a committed TensorFlow baseline.
- Support for Python 3.10–3.14.

### Changed

- `save()` requires `overwrite=True` before replacing an existing directory.
- Dropped Python 3.8 and 3.9.

### Fixed

- Random forest trees in the ensemble are now distinct from each other.
- `tree_to_nn_weights()` no longer ignores `seed=0`.
- `predict()` is deterministic when dropout is enabled.
- `find_optimal_epochs()` no longer crashes when `max_training_epochs <= 200`.
- Trees too shallow to build a hidden layer raise a clear error.

[Unreleased]: https://github.com/bwhewe-13/DJINN/compare/v1.1.2...HEAD
[1.1.2]: https://github.com/bwhewe-13/DJINN/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/bwhewe-13/DJINN/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/bwhewe-13/DJINN/releases/tag/v1.1.0
