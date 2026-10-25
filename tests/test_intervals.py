"""
tests/test_intervals.py — calibrate() and predict_interval().

    (pt-djinn) $ pytest tests/test_intervals.py -v
"""

import pickle

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from djinn import DJINN_Regressor, djinn


def fit_model(X, y):
    """Fit a small regressor quickly.

    Returns
    -------
    DJINN_Regressor
        Fitted model.
    """
    model = DJINN_Regressor(
        n_trees=2, max_tree_depth=4, learning_rate=0.01, epochs=20, random_state=0
    )
    return model.fit(X, y)


@pytest.fixture(scope="module")
def noisy_data():
    """Provide fit, calibration and test splits of a noisy regression problem.

    Returns
    -------
    tuple
        ``(X_fit, y_fit, X_cal, y_cal, X_test, y_test)``.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (1400, 3))
    y = X[:, 0] + np.sin(2 * X[:, 1]) + rng.normal(0, 0.3, len(X))
    return X[:400], y[:400], X[400:800], y[400:800], X[800:], y[800:]


@pytest.fixture(scope="module")
def calibrated(noisy_data):
    """Provide a model calibrated on the calibration split.

    Returns
    -------
    DJINN_Regressor
        Fitted and calibrated model.
    """
    X_fit, y_fit, X_cal, y_cal, _, _ = noisy_data
    return fit_model(X_fit, y_fit).calibrate(X_cal, y_cal)


class TestCalibrate:
    """Storing the calibration errors."""

    def test_returns_self_and_stores_sorted_errors(self, noisy_data):
        """Verify calibrate stores one sorted column of errors per output."""
        X_fit, y_fit, X_cal, y_cal, _, _ = noisy_data
        model = fit_model(X_fit, y_fit)
        assert model.calibrate(X_cal, y_cal) is model
        residuals = model.calibration_residuals_
        assert residuals.shape == (len(y_cal), 1)
        assert np.all(np.diff(residuals[:, 0]) >= 0)

    def test_requires_fit(self, noisy_data):
        """Verify calibrating an unfitted model raises NotFittedError."""
        _, _, X_cal, y_cal, _, _ = noisy_data
        with pytest.raises(NotFittedError):
            DJINN_Regressor().calibrate(X_cal, y_cal)

    def test_refit_clears_calibration(self, noisy_data):
        """Verify fitting again drops the stale calibration."""
        X_fit, y_fit, X_cal, y_cal, _, _ = noisy_data
        model = fit_model(X_fit, y_fit).calibrate(X_cal, y_cal)
        model.fit(X_fit, y_fit)
        assert not hasattr(model, "calibration_residuals_")


class TestPredictInterval:
    """Coverage, shape and error checks for the intervals."""

    @pytest.mark.parametrize("coverage", [0.5, 0.8, 0.9])
    def test_coverage_on_held_out_data(self, calibrated, noisy_data, coverage):
        """Verify the intervals contain about ``coverage`` of the test values."""
        _, _, _, _, X_test, y_test = noisy_data
        lower, upper = calibrated.predict_interval(X_test, coverage=coverage)
        inside = np.mean((y_test >= lower) & (y_test <= upper))
        assert abs(inside - coverage) < 0.07

    def test_shape_matches_predict(self, calibrated, noisy_data):
        """Verify the bounds have the shape of predict and surround it."""
        _, _, _, _, X_test, _ = noisy_data
        lower, upper = calibrated.predict_interval(X_test)
        pred = calibrated.predict(X_test)
        assert lower.shape == upper.shape == pred.shape
        assert np.all(lower <= pred) and np.all(pred <= upper)

    def test_wider_for_higher_coverage(self, calibrated, noisy_data):
        """Verify a higher coverage gives wider intervals."""
        _, _, _, _, X_test, _ = noisy_data
        lo50, hi50 = calibrated.predict_interval(X_test, coverage=0.5)
        lo90, hi90 = calibrated.predict_interval(X_test, coverage=0.9)
        assert np.all(hi90 - lo90 > hi50 - lo50)

    def test_multi_output(self, noisy_data):
        """Verify each output gets its own interval width."""
        X_fit, y_fit, X_cal, y_cal, X_test, _ = noisy_data
        model = fit_model(X_fit, np.column_stack([y_fit, 10 * y_fit]))
        model.calibrate(X_cal, np.column_stack([y_cal, 10 * y_cal]))
        lower, upper = model.predict_interval(X_test)
        assert lower.shape == (len(X_test), 2)
        width = (upper - lower)[0]
        assert width[1] > 5 * width[0]

    def test_requires_calibration(self, noisy_data):
        """Verify predict_interval raises before calibrate is called."""
        X_fit, y_fit, _, _, X_test, _ = noisy_data
        with pytest.raises(NotFittedError):
            fit_model(X_fit, y_fit).predict_interval(X_test)

    @pytest.mark.parametrize("coverage", [0, 1, 1.5])
    def test_rejects_bad_coverage(self, calibrated, noisy_data, coverage):
        """Verify coverage outside (0, 1) raises ValueError."""
        _, _, _, _, X_test, _ = noisy_data
        with pytest.raises(ValueError, match="between 0 and 1"):
            calibrated.predict_interval(X_test, coverage=coverage)

    def test_too_few_calibration_points(self, noisy_data):
        """Verify a coverage the calibration set can't support raises."""
        X_fit, y_fit, X_cal, y_cal, X_test, _ = noisy_data
        model = fit_model(X_fit, y_fit).calibrate(X_cal[:5], y_cal[:5])
        with pytest.raises(ValueError, match="at least 9"):
            model.predict_interval(X_test, coverage=0.9)
        model.calibrate(X_cal[:9], y_cal[:9])
        model.predict_interval(X_test, coverage=0.9)


class TestPersistence:
    """Keeping the calibration through save, load and pickle."""

    def test_save_and_load(self, calibrated, noisy_data, tmp_path):
        """Verify the calibration survives save and load."""
        _, _, _, _, X_test, _ = noisy_data
        calibrated.save(tmp_path / "model")
        loaded = djinn.load(tmp_path / "model")
        for before, after in zip(
            calibrated.predict_interval(X_test), loaded.predict_interval(X_test)
        ):
            np.testing.assert_allclose(before, after, rtol=1e-5)

    def test_pickle(self, calibrated, noisy_data):
        """Verify the calibration survives pickling."""
        _, _, _, _, X_test, _ = noisy_data
        loaded = pickle.loads(pickle.dumps(calibrated))
        np.testing.assert_array_equal(
            calibrated.predict_interval(X_test)[0], loaded.predict_interval(X_test)[0]
        )
