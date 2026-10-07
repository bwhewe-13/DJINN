"""
tests/test_sklearn.py — scikit-learn estimator compatibility.

    (pt-djinn) $ pytest tests/test_sklearn.py -v
"""

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.datasets import load_iris

from djinn import DJINN_Classifier, DJINN_Regressor


@pytest.fixture(scope="module")
def reg_data():
    """Provide a small single-output regression dataset.

    Returns
    -------
    tuple
        ``(X, y)`` with ``y`` of shape ``(n_samples, 1)``.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4))
    y = (X[:, 0] + X[:, 1] ** 2).reshape(-1, 1)
    return X, y


@pytest.fixture(scope="module")
def iris():
    """Provide the iris classification dataset.

    Returns
    -------
    tuple
        ``(X, y)`` with integer labels 0-2.
    """
    return load_iris(return_X_y=True)


class TestEstimatorBasics:
    """Parameter handling expected of every scikit-learn estimator."""

    @pytest.mark.parametrize("cls", [DJINN_Regressor, DJINN_Classifier])
    def test_get_params_round_trip(self, cls):
        """Verify constructor arguments come back unchanged from get_params."""
        model = cls(n_trees=3, max_tree_depth=5, dropout_keep_prob=0.9)
        params = model.get_params()
        assert params["n_trees"] == 3
        assert params["max_tree_depth"] == 5
        assert params["dropout_keep_prob"] == 0.9
        assert cls(**params).get_params() == params

    @pytest.mark.parametrize("cls", [DJINN_Regressor, DJINN_Classifier])
    def test_clone_copies_params(self, cls):
        """Verify clone builds an unfitted copy with the same parameters."""
        model = cls(n_trees=2).set_params(max_tree_depth=3)
        copy = clone(model)
        assert copy is not model
        assert copy.get_params() == model.get_params()

    def test_estimator_types(self):
        """Verify scikit-learn recognizes each model's estimator type."""
        assert is_regressor(DJINN_Regressor())
        assert is_classifier(DJINN_Classifier())

    def test_train_returns_self_and_keeps_params(self, reg_data):
        """Verify training returns the model and leaves parameters untouched."""
        X, y = reg_data
        model = DJINN_Regressor(n_trees=1)
        out = model.train(X, y, ntrees=2, epochs=2)
        assert out is model
        assert model.n_trees == 1
        assert model.n_trees_ == 2
        assert len(model.models_) == 2

    def test_refit_uses_new_data_scale(self, reg_data):
        """Verify a second fit rescales to the new data instead of reusing."""
        X, y = reg_data
        model = DJINN_Regressor().train(X, y, epochs=2)
        model.train(X * 10, y * 10, epochs=2)
        np.testing.assert_allclose(model.xscale_.data_max_, (X * 10).max(axis=0))
        np.testing.assert_allclose(model.yscale_.data_max_, (y * 10).max(axis=0))

    def test_classifier_score_is_accuracy(self, iris):
        """Verify the classifier's score() is accuracy on its predictions."""
        X, y = iris
        model = DJINN_Classifier().train(X, y, epochs=2)
        assert model.score(X, y) == pytest.approx(np.mean(model.predict(X) == y))


class TestFit:
    """fit() takes its settings from the constructor."""

    def test_fit_uses_constructor_settings(self, reg_data):
        """Verify fit returns self and trains for the configured epochs."""
        X, y = reg_data
        model = DJINN_Regressor(learning_rate=0.01, epochs=3, random_state=0)
        assert model.fit(X, y) is model
        assert len(model.nninfo["train_cost"]) == 3

    def test_fit_and_train_write_no_files(self, reg_data, tmp_path, monkeypatch):
        """Verify neither fit nor train writes to the working directory."""
        X, y = reg_data
        monkeypatch.chdir(tmp_path)
        DJINN_Regressor(learning_rate=0.01, epochs=2).fit(X, y)
        DJINN_Regressor().train(X, y, epochs=2)
        assert list(tmp_path.iterdir()) == []

    def test_random_state_makes_fit_reproducible(self, reg_data):
        """Verify two fits with the same random_state predict the same."""
        X, y = reg_data
        params = dict(n_trees=2, learning_rate=0.01, epochs=3, random_state=4)
        a = DJINN_Regressor(**params).fit(X, y).predict(X)
        b = DJINN_Regressor(**params).fit(X, y).predict(X)
        np.testing.assert_allclose(a, b)

    def test_old_fit_keywords_warn(self, reg_data):
        """Verify training options passed to fit still work, with a warning."""
        X, y = reg_data
        model = DJINN_Regressor()
        with pytest.warns(FutureWarning, match="deprecated"):
            model.fit(X, y, learning_rate=0.01, epochs=2, save_model=False)
        assert len(model.nninfo["train_cost"]) == 2

    def test_unknown_fit_keyword_raises(self, reg_data):
        """Verify a misspelled fit keyword is not silently ignored."""
        X, y = reg_data
        with pytest.raises(TypeError, match="learning_rte"):
            DJINN_Regressor().fit(X, y, learning_rte=0.01)
