"""
tests/test_clip_inputs.py — the clip_inputs option.

    (pt-djinn) $ pytest tests/test_clip_inputs.py -v
"""

import numpy as np
import pytest

from djinn import DJINN_Classifier, DJINN_Regressor, djinn


@pytest.fixture(scope="module")
def reg_data():
    """Provide a small regression dataset on [-1, 1].

    Returns
    -------
    tuple
        ``(X, y)`` with ``y`` of shape ``(n_samples,)``.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (200, 3))
    y = X[:, 0] + X[:, 1] ** 2
    return X, y


@pytest.fixture(scope="module")
def fitted(reg_data):
    """Provide a regressor fit without clipping.

    Returns
    -------
    DJINN_Regressor
        Fitted model.
    """
    X, y = reg_data
    model = DJINN_Regressor(
        n_trees=2, max_tree_depth=4, learning_rate=0.01, epochs=20, random_state=0
    )
    return model.fit(X, y)


class TestClipInputs:
    """Clipping inputs to the training range at prediction time."""

    def test_off_by_default(self, fitted, reg_data):
        """Verify inputs outside the training range aren't clipped by default."""
        assert fitted.clip_inputs is False
        far = np.array([[100.0, 0.0, 0.0]])
        edge = np.array([[reg_data[0][:, 0].max(), 0.0, 0.0]])
        assert not np.allclose(fitted.predict(far), fitted.predict(edge))

    def test_clipped_matches_edge_of_range(self, fitted, reg_data):
        """Verify a far-out input predicts the same as the training range edge."""
        X, _ = reg_data
        fitted.set_params(clip_inputs=True)
        try:
            far = np.array([[100.0, -100.0, 0.5]])
            edge = np.array([[X[:, 0].max(), X[:, 1].min(), 0.5]])
            np.testing.assert_allclose(fitted.predict(far), fitted.predict(edge))
        finally:
            fitted.set_params(clip_inputs=False)

    def test_no_change_inside_range(self, fitted, reg_data):
        """Verify clipping leaves predictions on training inputs unchanged."""
        X, _ = reg_data
        before = fitted.predict(X)
        fitted.set_params(clip_inputs=True)
        try:
            np.testing.assert_array_equal(fitted.predict(X), before)
        finally:
            fitted.set_params(clip_inputs=False)

    def test_classifier(self, reg_data):
        """Verify the classifier accepts clip_inputs and clips its inputs."""
        X, y = reg_data
        labels = (y > np.median(y)).astype(int)
        model = DJINN_Classifier(
            n_trees=1,
            learning_rate=0.01,
            epochs=10,
            clip_inputs=True,
            random_state=0,
        ).fit(X, labels)
        far = np.array([[100.0, 0.0, 0.0]])
        edge = np.array([[X[:, 0].max(), 0.0, 0.0]])
        np.testing.assert_allclose(model.predict_proba(far), model.predict_proba(edge))

    def test_kept_by_save_and_load(self, reg_data, tmp_path):
        """Verify save and load keep the setting."""
        X, y = reg_data
        model = DJINN_Regressor(
            n_trees=1, learning_rate=0.01, epochs=5, clip_inputs=True, random_state=0
        ).fit(X, y)
        model.save(tmp_path / "model")
        assert djinn.load(tmp_path / "model").clip_inputs is True
