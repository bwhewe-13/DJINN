"""
tests/test_unit_shared.py — Tests that must pass in BOTH environments.

These only test behaviour that both the TF and PyTorch implementations
are expected to share: the DJINN_Regressor class exists, predict() returns
the right shape and type, get_hyperparameters() returns a dict, and the
module-level load() function is present.

Run in either venv:
    (tf-djinn) $ pytest tests/test_unit_shared.py -v
    (pt-djinn) $ pytest tests/test_unit_shared.py -v
"""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from djinn import djinn


def make_model():
    """Construct a default DJINN regressor for shared tests.

    Returns
    -------
    DJINN_Regressor
        Fresh model instance.
    """
    return djinn.DJINN_Regressor()


def fit(model, X, y):
    """Train using whichever method the active backend exposes.

    Parameters
    ----------
    model : DJINN_Regressor
        Model instance to train.
    X : numpy.ndarray
        Training features.
    y : numpy.ndarray
        Training targets.

    Returns
    -------
    None
        Trains the model in place.
    """
    if hasattr(model, "train"):
        model.train(X, y)
    else:
        model.fit(X, y)


@pytest.fixture(scope="module")
def small_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 4))
    y = (X[:, 0] + X[:, 1] ** 2).reshape(-1, 1)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def multiout_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 4))
    y = np.column_stack([X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]])
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def trained_model(small_data):
    X_train, _, y_train, _ = small_data
    model = make_model()
    fit(model, X_train, y_train)
    return model


class TestSharedAPI:
    """Shared API surface checks required by both backends."""

    def test_regressor_class_exists(self):
        """Verify module exposes ``DJINN_Regressor``.

        Returns
        -------
        None
            Assertion-based test.
        """
        assert hasattr(
            djinn, "DJINN_Regressor"
        ), "djinn module must expose DJINN_Regressor"

    def test_predict_method_exists(self):
        """Verify ``DJINN_Regressor`` exposes ``predict``.

        Returns
        -------
        None
            Assertion-based test.
        """
        assert callable(
            getattr(make_model(), "predict", None)
        ), "DJINN_Regressor must have predict()"

    def test_get_hyperparameters_method_exists(self):
        """Verify ``DJINN_Regressor`` exposes ``get_hyperparameters``.

        Returns
        -------
        None
            Assertion-based test.
        """
        assert callable(
            getattr(make_model(), "get_hyperparameters", None)
        ), "DJINN_Regressor must have get_hyperparameters()"

    def test_module_level_load_exists(self):
        """Verify module exposes top-level ``load``.

        Returns
        -------
        None
            Assertion-based test.
        """
        assert callable(
            getattr(djinn, "load", None)
        ), "djinn module must expose a top-level load() function"


class TestOutputShapes:
    """Common output shape and dtype checks for prediction APIs."""

    def test_predict_row_count_matches_input(self, trained_model, small_data):
        """Verify prediction rows match the number of input rows.

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
        preds = trained_model.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]

    def test_predict_returns_numpy_array(self, trained_model, small_data):
        """Verify ``predict`` returns a NumPy array.

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
        preds = trained_model.predict(X_test)
        assert isinstance(
            preds, np.ndarray
        ), f"predict() must return np.ndarray, got {type(preds)}"

    def test_predict_dtype_is_float(self, trained_model, small_data):
        """Verify prediction array dtype is floating-point.

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
        preds = trained_model.predict(X_test)
        assert np.issubdtype(
            preds.dtype, np.floating
        ), f"predict() must return a float array, got dtype={preds.dtype}"

    def test_predict_multi_output_shape(self, multiout_data):
        """Verify prediction shape for a two-output regression task.

        Parameters
        ----------
        multiout_data : tuple
            Fixture providing train/test splits with two outputs.

        Returns
        -------
        None
            Assertion-based test.
        """
        X_train, X_test, y_train, _ = multiout_data
        model = make_model()
        fit(model, X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == (
            X_test.shape[0],
            2,
        ), f"Expected shape ({X_test.shape[0]}, 2), got {preds.shape}"


class TestHyperparameters:
    """Shared hyperparameter contract checks for both implementations."""

    def test_returns_dict(self, small_data):
        """Verify ``get_hyperparameters`` returns a dictionary.

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
        assert isinstance(
            params, dict
        ), f"get_hyperparameters() must return dict, got {type(params)}"

    def test_dict_is_not_empty(self, small_data):
        """Verify hyperparameter dictionary is non-empty.

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
        assert len(params) > 0, "get_hyperparameters() must return a non-empty dict"

    def test_epochs_value_is_positive(self, small_data):
        """Verify epoch count key exists and its value is positive.

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
        # Key may be "epochs" or "training_epochs" depending on implementation
        epoch_val = params.get("epochs") or params.get("training_epochs")
        assert (
            epoch_val is not None
        ), "Hyperparameters must contain an epochs key ('epochs' or 'training_epochs')"
        assert epoch_val >= 1, f"Epoch count must be >= 1, got {epoch_val}"


class TestEdgeCases:
    """Shared inference edge-case checks for prediction outputs."""

    def test_single_sample_prediction(self, trained_model, small_data):
        """Verify prediction works for a single input sample.

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
        preds = trained_model.predict(X_test[[0]])
        assert preds.shape[0] == 1

    def test_no_nan_in_predictions(self, trained_model, small_data):
        """Verify prediction outputs do not include NaN values.

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
        assert not np.any(
            np.isnan(trained_model.predict(X_test))
        ), "Predictions must not contain NaN"

    def test_no_inf_in_predictions(self, trained_model, small_data):
        """Verify prediction outputs do not include infinite values.

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
        assert not np.any(
            np.isinf(trained_model.predict(X_test))
        ), "Predictions must not contain Inf"

    def test_predict_on_training_data_does_not_raise(self, small_data):
        """Verify ``predict`` executes on training inputs without errors.

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
        fit(model, X_train, y_train)
        model.predict(X_train)
