"""
tests/test_sklearn.py — scikit-learn estimator compatibility.

    (pt-djinn) $ pytest tests/test_sklearn.py -v
"""

import json
import pickle

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from djinn import DJINN_Classifier, DJINN_Regressor, djinn


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


class TestInputValidation:
    """Inputs are checked the way scikit-learn estimators check them."""

    @pytest.mark.parametrize("cls", [DJINN_Regressor, DJINN_Classifier])
    def test_predict_before_fit_raises(self, cls, reg_data):
        """Verify predicting with an unfitted model raises NotFittedError."""
        X, _ = reg_data
        with pytest.raises(NotFittedError):
            cls().predict(X)

    def test_one_dimensional_X_raises(self, reg_data):
        """Verify a 1-D feature array is rejected instead of ignored."""
        X, y = reg_data
        with pytest.raises(ValueError, match="2D array"):
            DJINN_Regressor().train(X[:, 0], y, epochs=2)

    def test_wrong_feature_count_raises(self, reg_data):
        """Verify predict rejects data with a different number of features."""
        X, y = reg_data
        model = DJINN_Regressor().train(X, y, epochs=2)
        assert model.n_features_in_ == X.shape[1]
        with pytest.raises(ValueError, match="features"):
            model.predict(X[:, :2])

    def test_nan_input_raises(self, reg_data):
        """Verify NaN features are rejected before training."""
        X, y = reg_data
        X = X.copy()
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            DJINN_Regressor().train(X, y, epochs=2)

    def test_lists_are_accepted(self, reg_data):
        """Verify plain Python lists work for training and prediction."""
        X, y = reg_data
        model = DJINN_Regressor().train(X.tolist(), y.tolist(), epochs=2)
        assert model.predict(X[:5].tolist()).shape[0] == 5

    def test_dataframe_feature_names(self, reg_data):
        """Verify DataFrame column names are recorded."""
        pd = pytest.importorskip("pandas")
        X, y = reg_data
        df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        model = DJINN_Regressor().train(df, y, epochs=2)
        assert list(model.feature_names_in_) == ["a", "b", "c", "d"]
        assert model.predict(df).shape[0] == len(df)


class TestPredictShape:
    """predict() output follows the shape of the training target."""

    def test_one_dimensional_target(self, reg_data, tmp_path):
        """Verify a 1-D target gives 1-D predictions, also after reloading."""
        X, y = reg_data
        model = DJINN_Regressor().train(X, y.ravel(), epochs=2)
        assert model.predict(X).shape == (len(X),)
        model.save(tmp_path / "model")
        assert djinn.load(tmp_path / "model").predict(X).shape == (len(X),)

    def test_column_target_keeps_column(self, reg_data):
        """Verify an (n, 1) target still gives (n, 1) predictions."""
        X, y = reg_data
        model = DJINN_Regressor().train(X, y, epochs=2)
        assert model.predict(X).shape == (len(X), 1)


class TestVerbose:
    """Progress messages are opt-in."""

    def test_quiet_by_default(self, reg_data, capsys):
        """Verify fit prints nothing unless verbose is set."""
        X, y = reg_data
        DJINN_Regressor(learning_rate=0.01, epochs=2).fit(X, y)
        assert capsys.readouterr().out == ""

    def test_verbose_prints_progress(self, reg_data, tmp_path, capsys):
        """Verify verbose models report when trees are restored."""
        X, y = reg_data
        DJINN_Regressor().train(X, y, epochs=2).save(tmp_path / "model")
        model = djinn.load(tmp_path / "model")
        model.set_params(verbose=1).load_model(model.model_name, model.model_path)
        assert "Tree 0 restored" in capsys.readouterr().out


class TestClassifierLabels:
    """The classifier works with any label values and exposes probabilities."""

    def test_string_labels(self, iris):
        """Verify string labels are predicted back as the same strings."""
        X, y = iris
        names = np.array(["setosa", "versicolor", "virginica"])[y]
        model = DJINN_Classifier().train(X, names, epochs=2)
        assert list(model.classes_) == ["setosa", "versicolor", "virginica"]
        assert set(model.predict(X)) <= set(model.classes_)

    def test_labels_not_starting_at_zero(self, iris):
        """Verify labels 1-3 train and predict without index errors."""
        X, y = iris
        model = DJINN_Classifier().train(X, y + 1, epochs=2)
        np.testing.assert_array_equal(model.classes_, [1, 2, 3])
        assert set(model.predict(X)) <= {1, 2, 3}

    def test_predict_proba_matches_predict(self, iris):
        """Verify probabilities sum to 1 and their argmax is the prediction."""
        X, y = iris
        model = DJINN_Classifier(n_trees=3).train(X, y, epochs=5)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            model.classes_[proba.argmax(axis=1)], model.predict(X)
        )


class TestSaveLoad:
    """Saved models come back as the same kind of estimator."""

    def test_classifier_round_trip(self, iris, tmp_path):
        """Verify a saved classifier reloads with its labels and outputs."""
        X, y = iris
        names = np.array(["a", "b", "c"])[y]
        model = DJINN_Classifier(n_trees=2).train(X, names, epochs=3)
        model.save(tmp_path / "clf")
        loaded = djinn.load(tmp_path / "clf")
        assert isinstance(loaded, DJINN_Classifier)
        np.testing.assert_array_equal(loaded.classes_, model.classes_)
        np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
        np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))

    def test_old_classifier_file_loads(self, iris, tmp_path):
        """Verify a classifier saved without type or labels still loads."""
        X, y = iris
        DJINN_Classifier().train(X, y, epochs=2).save(tmp_path / "clf")
        json_path = tmp_path / "clf.json"
        state = json.loads(json_path.read_text())
        del state["estimator"], state["classes"]
        json_path.write_text(json.dumps(state))
        loaded = djinn.load(tmp_path / "clf")
        assert isinstance(loaded, DJINN_Classifier)
        np.testing.assert_array_equal(loaded.classes_, [0, 1, 2])
        assert loaded.predict(X).shape == (len(X),)


class TestIntegration:
    """DJINN works inside scikit-learn's model selection tools."""

    def test_pipeline(self, reg_data):
        """Verify DJINN trains and predicts as the last step of a Pipeline."""
        X, y = reg_data
        pipe = make_pipeline(
            StandardScaler(), DJINN_Regressor(learning_rate=0.01, epochs=5)
        )
        pipe.fit(X, y.ravel())
        assert pipe.predict(X).shape == (len(X),)

    def test_cross_val_score_in_parallel(self, reg_data):
        """Verify cross-validation runs, including across worker processes."""
        X, y = reg_data
        model = DJINN_Regressor(learning_rate=0.01, epochs=5, random_state=0)
        scores = cross_val_score(model, X, y.ravel(), cv=3, n_jobs=2)
        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))

    def test_grid_search(self, iris):
        """Verify GridSearchCV can tune the tree depth of a classifier."""
        X, y = iris
        model = DJINN_Classifier(learning_rate=0.01, epochs=5, random_state=0)
        search = GridSearchCV(model, {"max_tree_depth": [3, 4]}, cv=3)
        search.fit(X, y)
        assert search.best_params_["max_tree_depth"] in (3, 4)
        assert isinstance(search.best_estimator_, DJINN_Classifier)

    @pytest.mark.parametrize("cls", [DJINN_Regressor, DJINN_Classifier])
    def test_pickle_round_trip(self, cls, reg_data, iris):
        """Verify a pickled model predicts the same after unpickling."""
        X, y = iris if cls is DJINN_Classifier else reg_data
        model = cls(n_trees=2).train(X, y, epochs=3)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))
