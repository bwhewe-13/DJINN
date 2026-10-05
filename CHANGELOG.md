# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- `get_hyperparameters(seed=...)` is now reproducible. The learning-rate
  search drew its initial biases from OS entropy and ran dropout on the
  unseeded global torch RNG, so the same seed could select a different
  learning rate on each call.

### Changed

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

[Unreleased]: https://github.com/bwhewe-13/DJINN/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/bwhewe-13/DJINN/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/bwhewe-13/DJINN/releases/tag/v1.1.0
