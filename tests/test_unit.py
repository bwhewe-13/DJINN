"""
tests/test_unit.py — Unit tests specific to the PyTorch DJINN fork.

Tests here cover methods and behaviour that exist ONLY in the PyTorch fork
(train(), bma_predict(), save(), ntrees/learning_rate hyperparameter keys).
Tests that must pass in both implementations live in test_unit_shared.py.

    (pt-djinn) $ pytest tests/test_unit.py -v
"""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from djinn import djinn


def make_model():
    return djinn.DJINN_Regressor()


def train(model, X, y, ntrees=1, epochs=5, seed=None):
    """Train a model with optional random-state control for reproducibility."""
    kwargs = dict(ntrees=ntrees, epochs=epochs)
    if seed is not None:
        kwargs["random_state"] = seed
    model.train(X, y, **kwargs)


@pytest.fixture(scope="module")
def small_data():
    """80-sample, 4-feature regression dataset for fast tests."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 4))
    y = (X[:, 0] + X[:, 1] ** 2).reshape(-1, 1)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def multiout_data():
    """80-sample, 4-feature, 2-output dataset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 4))
    y = np.column_stack([X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]])
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def trained_model(small_data):
    """Return a trained model reused across tests without custom training."""
    X_train, _, y_train, _ = small_data
    model = make_model()
    train(model, X_train, y_train, ntrees=1, epochs=5, seed=0)
    return model


class TestPyTorchAPI:
    """Methods added in the PyTorch fork that don't exist in TF."""

    def test_train_method_exists(self):
        """Verify the PyTorch API exposes ``train``.

        Returns
        -------
        None
            Assertion-based test.
        """
        assert callable(
            getattr(make_model(), "train", None)
        ), "PyTorch fork must expose train() (not just fit())"

    def test_train_accepts_ntrees(self, small_data):
        """Verify ``train`` accepts the ``ntrees`` argument.

        Parameters
        ----------
        small_data : tuple
            Fixture providing train/test splits.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, _, y_train, _ = small_data
        model = make_model()
        train(model, X_train, y_train, ntrees=2, epochs=2)  # must not raise

    def test_bma_predict_method_exists(self):
        """Verify the PyTorch API exposes ``bma_predict``.

        Returns
        -------
        None
            Assertion-based test.
        """
        assert callable(
            getattr(make_model(), "bma_predict", None)
        ), "PyTorch fork must have bma_predict()"

    def test_save_method_exists(self):
        """Verify the PyTorch API exposes ``save``.

        Returns
        -------
        None
            Assertion-based test.
        """
        assert callable(
            getattr(make_model(), "save", None)
        ), "PyTorch fork must have save()"


class TestBMAShapes:
    """Shape and sampling checks for Bayesian model averaging outputs."""

    def test_bma_predict_returns_array_or_dict(self, trained_model, small_data):
        """Verify ``bma_predict`` returns a supported container type.

        Parameters
        ----------
        trained_model : DJINN_Regressor
            Trained model fixture.
        small_data : tuple
            Fixture providing train/test splits.

        Returns
        -------
        None
            Assertion-based test.
        """
        _, X_test, _, _ = small_data
        result = trained_model.bma_predict(X_test, n_iters=3)
        assert isinstance(
            result, (np.ndarray, dict)
        ), f"bma_predict() must return ndarray or dict, got {type(result)}"

    def test_bma_predict_has_multiple_samples(self, small_data):
        """Verify BMA outputs include multiple stochastic samples.

        Parameters
        ----------
        small_data : tuple
            Fixture providing train/test splits.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, X_test, y_train, _ = small_data
        model = make_model()
        train(model, X_train, y_train, ntrees=2, epochs=3, seed=0)
        result = model.bma_predict(X_test, n_iters=5)
        samples = result["predictions"] if isinstance(result, dict) else result
        arr = np.array(samples)
        assert (
            arr.shape[0] > 1
        ), f"bma_predict() must return >1 sample, got shape {arr.shape}"


class TestDeterminism:
    """Same seed must produce bit-identical predictions within this env."""

    def test_same_seed_gives_same_predictions(self, small_data):
        """Verify repeated training with the same seed is deterministic.

        Parameters
        ----------
        small_data : tuple
            Fixture providing train/test splits.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, X_test, y_train, _ = small_data

        m1 = make_model()
        train(m1, X_train, y_train, ntrees=1, epochs=5, seed=7)

        m2 = make_model()
        train(m2, X_train, y_train, ntrees=1, epochs=5, seed=7)

        np.testing.assert_allclose(
            m1.predict(X_test),
            m2.predict(X_test),
            rtol=1e-4,
            err_msg="Identical seeds must produce identical predictions",
        )

    def test_different_seeds_give_different_predictions(self, small_data):
        """Verify different seeds produce different predictions.

        Parameters
        ----------
        small_data : tuple
            Fixture providing train/test splits.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, X_test, y_train, _ = small_data

        m1 = make_model()
        train(m1, X_train, y_train, ntrees=1, epochs=5, seed=0)

        m2 = make_model()
        train(m2, X_train, y_train, ntrees=1, epochs=5, seed=99)

        assert not np.allclose(
            m1.predict(X_test), m2.predict(X_test), atol=1e-6
        ), "Different seeds should (almost certainly) give different predictions"


class TestSaveLoad:
    """Persistence and load-time API checks for saved models."""

    def test_roundtrip_predictions_are_identical(self, small_data, tmp_path):
        """Verify prediction equivalence before and after save/load.

        Parameters
        ----------
        small_data : tuple
            Fixture providing train/test splits.
        tmp_path : pathlib.Path
            Temporary directory fixture.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, X_test, y_train, _ = small_data
        model = make_model()
        train(model, X_train, y_train, ntrees=1, epochs=3, seed=0)
        preds_before = model.predict(X_test)

        model.save(str(tmp_path / "test_model"))
        loaded = djinn.load(str(tmp_path / "test_model"))
        preds_after = loaded.predict(X_test)

        np.testing.assert_allclose(
            preds_before,
            preds_after,
            rtol=1e-5,
            err_msg="Reloaded model must produce identical predictions",
        )

    def test_loaded_model_has_predict(self, small_data, tmp_path):
        """Verify loaded models expose the ``predict`` method.

        Parameters
        ----------
        small_data : tuple
            Fixture providing train/test splits.
        tmp_path : pathlib.Path
            Temporary directory fixture.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, _, y_train, _ = small_data
        model = make_model()
        train(model, X_train, y_train, ntrees=1, epochs=2)
        model.save(str(tmp_path / "test_model"))
        loaded = djinn.load(str(tmp_path / "test_model"))
        assert callable(getattr(loaded, "predict", None))


class TestHyperparameters:
    """Validation checks for hyperparameter dictionary outputs."""

    def test_required_keys_present(self, small_data):
        """Verify expected hyperparameter keys are present.

        Parameters
        ----------
        small_data : tuple
            Fixture providing train/test splits.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, _, y_train, _ = small_data
        params = make_model().get_hyperparameters(X_train, y_train)
        for key in ("ntrees", "epochs", "learning_rate"):
            assert key in params, f"Missing key in hyperparameters: '{key}'"

    def test_ntrees_is_positive_int(self, small_data):
        """Verify ``ntrees`` is a positive integer.

        Parameters
        ----------
        small_data : tuple
            Fixture providing train/test splits.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, _, y_train, _ = small_data
        params = make_model().get_hyperparameters(X_train, y_train)
        assert isinstance(params["ntrees"], (int, np.integer)) and params["ntrees"] >= 1

    def test_epochs_is_positive_int(self, small_data):
        """Verify ``epochs`` is a positive integer.

        Parameters
        ----------
        small_data : tuple
            Fixture providing train/test splits.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, _, y_train, _ = small_data
        params = make_model().get_hyperparameters(X_train, y_train)
        assert isinstance(params["epochs"], (int, np.integer)) and params["epochs"] >= 1

    def test_learning_rate_in_valid_range(self, small_data):
        """Verify ``learning_rate`` lies in an expected numeric range.

        Parameters
        ----------
        small_data : tuple
            Fixture providing train/test splits.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, _, y_train, _ = small_data
        params = make_model().get_hyperparameters(X_train, y_train)
        lr = params["learning_rate"]
        assert 1e-6 <= lr <= 1.0, f"learning_rate={lr} is outside [1e-6, 1.0]"


class TestEdgeCases:
    """Edge-case behavior checks for PT-specific execution paths."""

    def test_predict_with_multiple_trees(self, small_data):
        """Verify prediction works with ``ntrees > 1``.

        Parameters
        ----------
        small_data : tuple
            Fixture providing train/test splits.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, X_test, y_train, _ = small_data
        model = make_model()
        train(model, X_train, y_train, ntrees=3, epochs=2)
        preds = model.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]

    def test_no_nan_with_multiple_trees(self, small_data):
        """Verify multi-tree predictions do not contain NaN values.

        Parameters
        ----------
        small_data : tuple
            Fixture providing train/test splits.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, X_test, y_train, _ = small_data
        model = make_model()
        train(model, X_train, y_train, ntrees=3, epochs=2)
        assert not np.any(np.isnan(model.predict(X_test)))
