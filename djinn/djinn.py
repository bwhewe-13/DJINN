###############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Originally written by K. Humbird (humbird1@llnl.gov), L. Peterson
# (peterson76@llnl.gov).
#
# PyTorch rewrite: Copyright (c) 2024-2026, Ben Whewell.
#
# LLNL-CODE-754815
#
# All rights reserved.
#
# This file is part of DJINN.
#
# For details, see github.com/LLNL/djinn.
#
# For details about use and distribution, please read DJINN/LICENSE .
###############################################################################

"""Public DJINN API for training, inference, and model persistence.

This module exposes the high-level regression and classification interfaces,
including hyperparameter selection, model training, Bayesian prediction, and
loading/saving of serialized DJINN models.
"""

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

# Functions from the provided modules
from djinn.neural_network import (
    get_hyperparams,
    load_tree_model,
    torch_continue_training,
    torch_dropout_regression,
)
from djinn.random_forest import fit_scalers, train_forest, tree_to_nn_weights


def _scaler_from_state(state):
    """Rebuild a fitted ``MinMaxScaler`` from its saved min/max values.

    Parameters
    ----------
    state : dict
        Dictionary with ``data_min_`` and ``data_max_`` lists.

    Returns
    -------
    MinMaxScaler
        Scaler ready for ``transform`` and ``inverse_transform``.
    """
    scaler = MinMaxScaler()
    scaler.data_min_ = np.array(state["data_min_"])
    scaler.data_max_ = np.array(state["data_max_"])
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.scale_ = np.divide(
        1.0,
        scaler.data_range_,
        out=np.zeros_like(scaler.data_range_, dtype=float),
        where=scaler.data_range_ != 0,
    )
    scaler.min_ = -scaler.data_min_ * scaler.scale_
    scaler.n_features_in_ = scaler.data_min_.shape[0]
    return scaler


def _scaler_state(scaler):
    """Return the JSON-serializable min/max values of a fitted scaler.

    Parameters
    ----------
    scaler : MinMaxScaler or None
        Fitted scaler.

    Returns
    -------
    dict or None
        ``data_min_`` and ``data_max_`` lists, or ``None`` without a scaler.
    """
    if scaler is None:
        return None
    return {
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
    }


_DEPRECATED_FIT_ARGS = {
    "epochs",
    "learning_rate",
    "learn_rate",
    "batch_size",
    "weight_decay",
    "save_files",
    "save_model",
    "model_name",
    "model_path",
    "seed",
}


class _DJINNBase(BaseEstimator):
    """Shared implementation for :class:`DJINN_Regressor` and
    :class:`DJINN_Classifier`.

    Parameters
    ----------
    n_trees : int, optional
        Number of trees in the random forest (equal to the number of
        neural networks).
    max_tree_depth : int, optional
        Maximum depth of decision tree. The neural network has
        ``max_tree_depth - 1`` hidden layers.
    dropout_keep_prob : float, optional
        Probability of keeping a neuron in dropout layers.
    epochs : int or None, optional
        Training epochs used by :meth:`fit`. ``None`` picks them
        automatically when ``learning_rate`` is also ``None``, otherwise
        uses 1000.
    learning_rate : float or None, optional
        Learning rate used by :meth:`fit`. ``None`` runs
        :meth:`get_hyperparameters` to choose it.
    batch_size : int or None, optional
        Minibatch size used by :meth:`fit`. ``None`` uses 5% of the samples.
    weight_decay : float, optional
        Multiplier for the L2 penalty on weights.
    random_state : int or None, optional
        Seed for the forest, weight initialization, and training.
    device : str or torch.device, optional
        Device used for training and inference.
    verbose : int, optional
        Print progress messages when greater than 0.
    """

    _regression = True

    def __init__(
        self,
        n_trees=1,
        max_tree_depth=4,
        dropout_keep_prob=1.0,
        *,
        epochs=None,
        learning_rate=None,
        batch_size=None,
        weight_decay=1.0e-8,
        random_state=None,
        device="cpu",
        verbose=0,
    ):
        self.n_trees = n_trees
        self.max_tree_depth = max_tree_depth
        self.dropout_keep_prob = dropout_keep_prob
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    # Old attribute names. Properties, since fit() may only add names ending in _
    @property
    def nninfo(self):
        """dict or None: Training history and weights from the last fit."""
        return getattr(self, "nninfo_", None)

    @property
    def model_name(self):
        """str or None: Name of the saved model directory."""
        return getattr(self, "model_name_", None)

    @model_name.setter
    def model_name(self, value):
        self.model_name_ = value

    @property
    def model_path(self):
        """str or None: Parent directory of the saved model."""
        return getattr(self, "model_path_", None)

    @model_path.setter
    def model_path(self, value):
        self.model_path_ = value

    def _torch_device(self):
        """Return :attr:`device` as a ``torch.device``.

        Returns
        -------
        torch.device
            Device used for training and inference.
        """
        return torch.device(self.device)

    def _fit_scalers(self, X, Y):
        """Fit MinMax scalers on raw data.

        Parameters
        ----------
        X : ndarray
            Raw input feature matrix of shape ``(n_samples, n_features)``.
        Y : ndarray
            Raw target array of shape ``(n_samples, n_outputs)``.

        Returns
        -------
        None
        """
        self.xscale_, self.yscale_ = fit_scalers(X, Y, self._regression)
        # Allow predictions outside the training range
        self.xscale_.clip = False

    def _validate_training_data(self, X, Y):
        """Check training data and record the input shape.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        Y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Training targets.

        Returns
        -------
        tuple[ndarray, ndarray]
            ``X`` and ``Y`` as arrays.
        """
        return validate_data(
            self,
            X,
            Y,
            multi_output=True,
            y_numeric=self._regression,
            dtype=np.float64,
            ensure_min_samples=2,
        )

    def _state(self, model_name, model_path):
        """Return the JSON-serializable state used to rebuild this model.

        Parameters
        ----------
        model_name : str
            Name of the model directory.
        model_path : str
            Parent directory of the model directory.

        Returns
        -------
        dict
            Hyperparameters, paths, and scaler values.
        """
        classes = getattr(self, "classes_", None)
        return {
            "estimator": type(self).__name__,
            "n_trees": self.n_trees_,
            "tree_max_depth": self.max_tree_depth,
            "dropout_keep_prob": self.dropout_keep_prob,
            "regression": self._regression,
            "model_name": model_name,
            "model_path": model_path,
            "xscale": _scaler_state(self.xscale_),
            "yscale": _scaler_state(self.yscale_),
            "y_1d": getattr(self, "_y_1d", False),
            "classes": None if classes is None else classes.tolist(),
        }

    def _save_json(self):
        """Save model metadata and scalers to a JSON sidecar file.

        Writes ``<model_name>.json`` in ``self.model_path`` for later
        reconstruction via :meth:`from_json`.

        Returns
        -------
        None
        """
        json_path = Path(self.model_path) / f"{self.model_name}.json"
        with open(json_path, "w") as f:
            json.dump(self._state(self.model_name, self.model_path), f, indent=2)

    def get_hyperparameters(self, X, Y, weight_decay=1.0e-8, seed=None):
        """Automatically select DJINN hyperparameters.

        Returns learning rate, number of epochs, and batch size by running
        a short auto-tuning search using the PyTorch training utilities in
        ``neural_network.py``.

        Parameters
        ----------
        X : ndarray
            Input feature matrix for training.
        Y : ndarray
            Target array for training.
        weight_decay : float, optional
            Multiplier for L2 penalty on weights.
        seed : int or None, optional
            Random seed for reproducibility. Defaults to
            :attr:`random_state`.

        Raises
        ------
        Exception
            If a decision tree cannot be built from the data.

        Returns
        -------
        dict
            Dictionary with keys ``batch_size``, ``learning_rate``, and
            ``epochs``.
        """
        if seed is None:
            seed = self.random_state
        X, Y = self._validate_training_data(X, Y)

        single_output = Y.ndim == 1
        if single_output:
            Y = Y.reshape(-1, 1)

        self._fit_scalers(X, Y)

        rfr = train_forest(
            X,
            Y,
            self.n_trees,
            self.max_tree_depth,
            self.xscale_,
            self.yscale_,
            self._regression,
            seed,
        )

        tree_to_network = tree_to_nn_weights(
            self._regression, X, Y, self.n_trees, rfr, seed
        )

        if self.verbose:
            print("Finding optimal hyper-parameters...")
        nn_batch_size, learning_rate, nn_epochs = get_hyperparams(
            self._regression,
            tree_to_network,
            self.xscale_,
            self.yscale_,
            X,
            Y,
            self.dropout_keep_prob,
            weight_decay,
            seed=seed,
            device=self._torch_device(),
            verbose=bool(self.verbose),
        )

        return {
            "batch_size": nn_batch_size,
            "learning_rate": learning_rate,
            # Backward-compatible alias used by older callers/tests.
            "learn_rate": learning_rate,
            "epochs": nn_epochs,
            "ntrees": self.n_trees,
        }

    def train(
        self,
        X,
        Y,
        epochs=1000,
        learning_rate=0.001,
        learn_rate=None,
        batch_size=0,
        weight_decay=1.0e-8,
        save_files=False,
        save_model=False,
        model_name="djinn_model",
        model_path="./",
        ntrees=None,
        seed=None,
        eval_every=1,
    ):
        """Train DJINN with specified hyperparameters.

        Builds a random forest, maps each tree to a PyTorch MLP via
        ``random_forest.tree_to_nn_weights``, then trains every network
        using ``neural_network.torch_dropout_regression``.

        Parameters
        ----------
        X : ndarray
            Input feature matrix for training.
        Y : ndarray
            Target array for training.
        epochs : int, optional
            Number of training epochs.
        learning_rate : float, optional
            Learning rate for weight and bias optimization.
        learn_rate : float or None, optional
            Backward-compatible alias for ``learning_rate``.
        batch_size : int, optional
            Number of samples per batch. If ``0``, uses 5% of the dataset.
        weight_decay : float, optional
            Multiplier for L2 penalty on weights.
        save_files : bool, optional
            If ``True``, saves train/validation cost per epoch and
            weights/biases.
        save_model : bool, optional
            If ``True``, saves the trained model.
        model_name : str, optional
            File name for the model when ``save_model`` is ``True``.
        model_path : str, optional
            Directory where model/files are saved.
        ntrees : int or None, optional
            Number of trees to train. Defaults to :attr:`n_trees`.
        seed : int or None, optional
            Random seed for reproducibility. Defaults to
            :attr:`random_state`.
        eval_every : int, optional
            Compute the validation loss every ``eval_every`` epochs.

        Raises
        ------
        Exception
            If a decision tree cannot be built from the data.

        Returns
        -------
        self
            The trained model.
        """
        if learn_rate is not None:
            learning_rate = learn_rate
        if seed is None:
            seed = self.random_state

        self.n_trees_ = int(ntrees) if ntrees is not None else self.n_trees
        self.model_name_ = model_name
        self.model_path_ = model_path

        X, Y = self._validate_training_data(X, Y)

        self._y_1d = Y.ndim == 1
        if self._y_1d:
            Y = Y.reshape(-1, 1)

        self._fit_scalers(X, Y)

        rfr = train_forest(
            X,
            Y,
            self.n_trees_,
            self.max_tree_depth,
            self.xscale_,
            self.yscale_,
            self._regression,
            seed,
        )

        tree_to_network = tree_to_nn_weights(
            self._regression, X, Y, self.n_trees_, rfr, seed
        )

        if batch_size == 0:
            batch_size = int(np.ceil(0.05 * len(Y)))

        self.nninfo_ = torch_dropout_regression(
            self._regression,
            tree_to_network,
            self.xscale_,
            self.yscale_,
            X,
            Y,
            ntrees=self.n_trees_,
            lr=learning_rate,
            n_epochs=epochs,
            batch_size=batch_size,
            dropout_keep_prob=self.dropout_keep_prob,
            weight_decay=weight_decay,
            # kwargs forwarded to torch_dropout_regression
            save_model=save_model,
            save_files=save_files,
            model_path=str(Path(model_path) / model_name),
            seed=seed,
            device=self._torch_device(),
            eval_every=eval_every,
        )

        # Keep the live models so predict() works without files on disk.
        self.models_ = self.nninfo_["models"]

        if save_model:
            saved_model_dir = self.nninfo_.get("model_dir")
            if saved_model_dir:
                saved_model_dir = Path(saved_model_dir)
                self.model_name_ = saved_model_dir.name
                self.model_path_ = str(saved_model_dir.parent)
            self._save_json()
        return self

    def fit(self, X, y, **kwargs):
        """Train DJINN using the settings given to the constructor.

        When :attr:`learning_rate` is ``None``, :meth:`get_hyperparameters`
        picks the learning rate, and also the epochs and batch size unless
        they were set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Training targets.
        **kwargs
            Deprecated. ``epochs``, ``learning_rate``, ``learn_rate``,
            ``batch_size``, ``weight_decay``, ``save_files``, ``save_model``,
            ``model_name``, ``model_path``, and ``seed`` are still accepted
            but will be removed in 2.0. Set them in the constructor, or use
            :meth:`train` and :meth:`save`.

        Returns
        -------
        self
            The trained model.
        """
        unknown = set(kwargs) - _DEPRECATED_FIT_ARGS
        if unknown:
            raise TypeError(f"fit() got unexpected keyword arguments {sorted(unknown)}")
        if kwargs:
            warnings.warn(
                "Passing training options to fit() is deprecated and will be "
                "removed in 2.0. Set them in the constructor, or use train() "
                "and save().",
                FutureWarning,
                stacklevel=2,
            )

        epochs = kwargs.get("epochs", self.epochs)
        learning_rate = kwargs.get(
            "learning_rate", kwargs.get("learn_rate", self.learning_rate)
        )
        batch_size = kwargs.get("batch_size", self.batch_size)
        weight_decay = kwargs.get("weight_decay", self.weight_decay)
        seed = kwargs.get("seed", self.random_state)

        if learning_rate is None:
            optimal = self.get_hyperparameters(X, y, weight_decay, seed)
            learning_rate = optimal["learning_rate"]
            if epochs is None:
                epochs = optimal["epochs"]
            if batch_size is None:
                batch_size = optimal["batch_size"]

        return self.train(
            X,
            y,
            epochs=1000 if epochs is None else epochs,
            learning_rate=learning_rate,
            batch_size=0 if batch_size is None else batch_size,
            weight_decay=weight_decay,
            save_files=kwargs.get("save_files", False),
            save_model=kwargs.get("save_model", False),
            model_name=kwargs.get("model_name", "djinn_model"),
            model_path=kwargs.get("model_path", "./"),
            seed=seed,
        )

    @classmethod
    def from_json(cls, json_path):
        """Reconstruct a model from a saved JSON state file.

        Restores all hyperparameters and scalers so the instance is ready
        for :meth:`load_model`, :meth:`predict`, or :meth:`continue_training`.

        Parameters
        ----------
        json_path : str or pathlib.Path
            Path to the ``.json`` file written by :meth:`train`.

        Returns
        -------
        DJINN_Regressor or DJINN_Classifier
            Restored model instance.
        """
        with open(json_path, "r") as f:
            state = json.load(f)

        obj = cls(
            n_trees=state["n_trees"],
            max_tree_depth=state["tree_max_depth"],
            dropout_keep_prob=state["dropout_keep_prob"],
        )
        obj.n_trees_ = state["n_trees"]
        obj.model_name_ = state["model_name"]
        obj.model_path_ = state["model_path"]
        obj.xscale_ = _scaler_from_state(state["xscale"])
        obj.xscale_.clip = False
        obj.n_features_in_ = obj.xscale_.n_features_in_
        obj._y_1d = state.get("y_1d", False)
        if state.get("classes") is not None:
            obj.classes_ = np.array(state["classes"])
        obj.yscale_ = (
            _scaler_from_state(state["yscale"]) if state["yscale"] is not None else None
        )
        return obj

    def load_model(self, model_name, model_path):
        """Reload PyTorch checkpoints for a saved model.

        Restores each tree's ``.pt`` checkpoint from disk using
        ``neural_network.load_tree_model``.

        Parameters
        ----------
        model_name : str
            Name of the saved model directory.
        model_path : str or pathlib.Path
            Parent directory that contains the model folder.

        Returns
        -------
        None
        """
        model_dir = Path(model_path) / model_name

        models = {}
        for tree_idx in range(self.n_trees_):
            checkpoint_path = model_dir / f"tree_{tree_idx}.pt"
            model, _ = load_tree_model(
                checkpoint_path,
                self._torch_device(),
                self.dropout_keep_prob,
                tree_idx,
                verbose=bool(self.verbose),
            )
            models[tree_idx] = model

        self.models_ = models

    def close_model(self):
        """Release all loaded PyTorch models from memory.

        Returns
        -------
        None
        """
        self.models_ = None

    def _tree_outputs(self, x_test, n_iters, seed, transform):
        """Run every tree network on ``x_test`` and collect its outputs.

        Parameters
        ----------
        x_test : ndarray
            Input feature matrix for testing.
        n_iters : int or None
            Number of forward passes per network. ``None`` runs a single
            deterministic pass with dropout disabled.
        seed : int or None
            Random seed for reproducibility.
        transform : callable
            Maps a raw network output tensor to a NumPy prediction array.

        Returns
        -------
        tuple[ndarray, dict]
            Stacked predictions with shape
            ``(n_iters * n_trees, n_test, n_outputs)`` and the raw sample
            dictionary with ``inputs`` and per-tree ``predictions``.
        """
        non_bayes = n_iters is None
        if non_bayes:
            n_iters = 1

        if seed is not None:
            torch.manual_seed(seed)

        check_is_fitted(self, "xscale_")
        x_test = validate_data(self, x_test, reset=False, dtype=np.float64)

        if getattr(self, "models_", None) is None:
            self.load_model(self.model_name, self.model_path)

        samples = {"inputs": x_test, "predictions": {}}

        device = self._torch_device()
        x_scaled = self.xscale_.transform(x_test)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)

        for tree_idx in range(self.n_trees_):
            model = self.models_[tree_idx].to(device)
            if non_bayes:
                model.eval()  # single deterministic pass, no dropout
            else:
                model.train()  # keep dropout active for Bayesian sampling

            with torch.no_grad():
                tree_preds = [transform(model(x_tensor)) for _ in range(n_iters)]
            samples["predictions"][f"tree{tree_idx}"] = tree_preds

        preds = self.collect_tree_predictions(samples["predictions"])
        return preds, samples

    def bayesian_predict(self, x_test, n_iters, seed=None):
        """Bayesian distribution of predictions for a set of test inputs.

        Evaluates each tree network ``n_iters`` times (with dropout active)
        to build a predictive distribution, then returns the 25th, 50th, and
        75th percentiles alongside the raw sample dictionary.

        Parameters
        ----------
        x_test : ndarray
            Input feature matrix for testing.
        n_iters : int or None
            Number of forward passes per network per test point.
            Pass ``None`` for a single non-Bayesian pass.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        ndarray or tuple
            If ``n_iters`` is ``None``, returns mean predictions with shape
            ``(n_test, n_outputs)``. Otherwise returns
            ``(lower, middle, upper, samples)``, where percentile arrays have
            shape ``(n_test, n_outputs)`` and ``samples`` contains per-tree
            prediction draws.
        """

        def to_targets(raw):
            return self.yscale_.inverse_transform(raw.cpu().double().numpy())

        preds, samples = self._tree_outputs(x_test, n_iters, seed, to_targets)

        if n_iters is None:
            return np.mean(preds, axis=0)

        middle = np.percentile(preds, 50, axis=0)
        lower = np.percentile(preds, 25, axis=0)
        upper = np.percentile(preds, 75, axis=0)
        return lower, middle, upper, samples

    def predict(self, x_test, seed=None):
        """Predict target values for a set of test inputs.

        Calls :meth:`bayesian_predict` with ``n_iters=None`` (single
        deterministic forward pass per network) and returns the mean.

        Parameters
        ----------
        x_test : ndarray
            Input feature matrix for testing.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        ndarray
            Mean target value prediction for each test point, shape
            ``(n_test,)`` if the model was trained on a 1-D target, otherwise
            ``(n_test, n_outputs)``.
        """
        preds = self.bayesian_predict(x_test, None, seed)
        if getattr(self, "_y_1d", False):
            preds = preds.ravel()
        return preds

    def bma_predict(self, x_test, n_iters=100, seed=None):
        """Return Bayesian model averaging samples and summary statistics.

        Parameters
        ----------
        x_test : ndarray
            Input feature matrix for testing.
        n_iters : int, optional
            Number of stochastic forward passes per tree.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            Dictionary containing percentile summaries and stacked prediction
            samples under ``predictions`` with shape
            ``(n_iters * n_trees, n_test, n_outputs)``.
        """
        lower, middle, upper, samples = self.bayesian_predict(x_test, n_iters, seed)
        preds = self.collect_tree_predictions(samples["predictions"])
        return {
            "lower": lower,
            "middle": middle,
            "upper": upper,
            "predictions": preds,
        }

    def save(self, model_path, overwrite=False):
        """Persist the currently loaded model under an explicit output path.

        Checkpoints are written from the in-memory models, so this works
        whether or not :meth:`train` was called with ``save_model=True``.

        Parameters
        ----------
        model_path : str or pathlib.Path
            Target base path. Writes checkpoints to ``<model_path>/`` and
            metadata to ``<model_path>.json``.
        overwrite : bool, optional
            If ``True``, delete and replace ``model_path`` when it already
            exists. Defaults to ``False``, which raises instead of silently
            deleting an existing directory.

        Returns
        -------
        pathlib.Path
            Saved model directory path.

        Raises
        ------
        RuntimeError
            If there are no trained or loaded models to save.
        FileExistsError
            If ``model_path`` already exists and ``overwrite`` is ``False``.
        """
        if not getattr(self, "models_", None):
            raise RuntimeError("No models to save. Call train() or load_model() first.")

        target = Path(model_path)
        target_dir = target
        target_json = target.with_suffix(".json")

        if target_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{target_dir} already exists. Pass overwrite=True to "
                    "replace it."
                )
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)

        for tree_idx, model in self.models_.items():
            layers = [*model.hidden_layers, model.output_layer]
            network_shape = [layers[0].in_features]
            network_shape += [layer.out_features for layer in layers]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "network_shape": network_shape,
                },
                target_dir / f"tree_{tree_idx}.pt",
            )

        state = self._state(target_dir.name, str(target_dir.parent))
        with open(target_json, "w") as f:
            json.dump(state, f, indent=2)

        return target_dir

    def collect_tree_predictions(self, predictions):
        """Gather and reshape the full distribution of per-tree predictions.

        Parameters
        ----------
        predictions : dict
            ``"predictions"`` sub-dictionary from the dictionary returned by
            :meth:`bayesian_predict`.

        Returns
        -------
        ndarray
            Reshaped predictions with shape
            ``(n_iters * n_trees, n_test, n_outputs)``.
        """
        n_out = predictions["tree0"][0].shape[1]
        n_iters = len(predictions["tree0"])
        x_length = predictions["tree0"][0].shape[0]
        preds = np.array([predictions[t] for t in predictions]).reshape(
            (n_iters * len(predictions), x_length, n_out)
        )
        return preds

    def continue_training(
        self,
        X,
        Y,
        training_epochs,
        learning_rate,
        batch_size,
        learn_rate=None,
        seed=None,
    ):
        """Continue training an existing model (must call :meth:`load_model` first).

        Delegates to ``neural_network.torch_continue_training`` and re-saves
        each tree checkpoint in place.

        Parameters
        ----------
        X : ndarray
            Input feature matrix for training.
        Y : ndarray
            Target array for training.
        training_epochs : int
            Additional epochs to train.
        learning_rate : float
            Learning rate.
        learn_rate : float or None, optional
            Backward-compatible alias for ``learning_rate``.
        batch_size : int
            Number of samples per batch.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        None
        """
        if learn_rate is not None:
            learning_rate = learn_rate

        model_dir = Path(self.model_path) / self.model_name

        torch_continue_training(
            regression=self._regression,
            xscale=self.xscale_,
            yscale=self.yscale_,
            x=X,
            y=Y,
            ntrees=self.n_trees_,
            lr=learning_rate,
            n_epochs=training_epochs,
            batch_size=batch_size,
            dropout_keep_prob=self.dropout_keep_prob,
            model_dir=model_dir,
            model_name=self.model_name,
            weight_decay=0.0,
            seed=seed,
            device=self._torch_device(),
            verbose=bool(self.verbose),
        )


class DJINN_Regressor(RegressorMixin, _DJINNBase):
    """DJINN regression model (PyTorch backend).

    A scikit-learn compatible regressor: it supports :func:`sklearn.base.clone`,
    :meth:`get_params`/:meth:`set_params`, and :meth:`score` (R²).

    Parameters
    ----------
    n_trees : int, optional
        Number of trees in the random forest (equal to the number of
        neural networks).
    max_tree_depth : int, optional
        Maximum depth of decision tree. The neural network has
        ``max_tree_depth - 1`` hidden layers.
    dropout_keep_prob : float, optional
        Probability of keeping a neuron in dropout layers.
    epochs : int or None, optional
        Training epochs used by :meth:`fit`. ``None`` picks them
        automatically when ``learning_rate`` is also ``None``, otherwise
        uses 1000.
    learning_rate : float or None, optional
        Learning rate used by :meth:`fit`. ``None`` runs
        :meth:`get_hyperparameters` to choose it.
    batch_size : int or None, optional
        Minibatch size used by :meth:`fit`. ``None`` uses 5% of the samples.
    weight_decay : float, optional
        Multiplier for the L2 penalty on weights.
    random_state : int or None, optional
        Seed for the forest, weight initialization, and training.
    device : str or torch.device, optional
        Device used for training and inference.
    verbose : int, optional
        Print progress messages when greater than 0.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags


class DJINN_Classifier(ClassifierMixin, _DJINNBase):
    """DJINN classification model.

    Shares training, saving, and loading with :class:`DJINN_Regressor`. The
    difference is in :meth:`bayesian_predict`, where no output scaling is
    applied and ``np.argmax`` converts softmax distributions into class
    predictions. :meth:`score` reports accuracy.

    Parameters
    ----------
    n_trees : int, optional
        Number of trees in the random forest (equal to the number of
        neural networks).
    max_tree_depth : int, optional
        Maximum depth of decision tree. The neural network has
        ``max_tree_depth - 1`` hidden layers.
    dropout_keep_prob : float, optional
        Probability of keeping a neuron in dropout layers.
    epochs : int or None, optional
        Training epochs used by :meth:`fit`. ``None`` picks them
        automatically when ``learning_rate`` is also ``None``, otherwise
        uses 1000.
    learning_rate : float or None, optional
        Learning rate used by :meth:`fit`. ``None`` runs
        :meth:`get_hyperparameters` to choose it.
    batch_size : int or None, optional
        Minibatch size used by :meth:`fit`. ``None`` uses 5% of the samples.
    weight_decay : float, optional
        Multiplier for the L2 penalty on weights.
    random_state : int or None, optional
        Seed for the forest, weight initialization, and training.
    device : str or torch.device, optional
        Device used for training and inference.
    verbose : int, optional
        Print progress messages when greater than 0.
    """

    _regression = False

    def _validate_training_data(self, X, Y):
        """Check training data and encode class labels as 0..n_classes-1.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        Y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        tuple[ndarray, ndarray]
            ``X`` as an array and the encoded labels.
        """
        X, Y = validate_data(self, X, Y, dtype=np.float64, ensure_min_samples=2)
        check_classification_targets(Y)
        self.classes_, Y = np.unique(Y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError(
                "DJINN_Classifier needs at least 2 classes, got 1 class: "
                f"{self.classes_[0]!r}"
            )
        return X, Y

    def _labels(self, indices):
        """Map class indices back to the original labels.

        Parameters
        ----------
        indices : ndarray
            Class indices.

        Returns
        -------
        ndarray
            Labels from :attr:`classes_`, or the indices for models saved
            without them.
        """
        classes = getattr(self, "classes_", None)
        return indices if classes is None else classes[indices]

    def _tree_probabilities(self, x_test, n_iters, seed):
        """Return softmax outputs from every tree network.

        Parameters
        ----------
        x_test : array-like of shape (n_test, n_features)
            Input features.
        n_iters : int or None
            Number of forward passes per network, or ``None`` for one
            deterministic pass.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        tuple[ndarray, dict]
            Probabilities with shape ``(n_iters * n_trees, n_test, n_classes)``
            and the raw sample dictionary.
        """

        def to_probabilities(logits):
            return torch.softmax(logits, dim=1).cpu().numpy()

        return self._tree_outputs(x_test, n_iters, seed, to_probabilities)

    def bayesian_predict(self, x_test, n_iters, seed=None):
        """Bayesian distribution of class predictions for a set of test inputs.

        Evaluates each tree network ``n_iters`` times (with dropout active)
        to build a predictive distribution over class probabilities, then
        returns the ``argmax`` of the 25th, 50th, and 75th percentiles as
        class labels alongside the raw sample dictionary.

        Parameters
        ----------
        x_test : ndarray
            Input feature matrix for testing.
        n_iters : int or None
            Number of forward passes per network per test point.
            Pass ``None`` for a single deterministic pass.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        ndarray or tuple
            If ``n_iters`` is ``None``, returns a 1-D array of predicted class
            labels with shape ``(n_test,)``. Otherwise returns
            ``(lower, middle, upper, samples)``, where percentile outputs are
            1-D arrays of class labels and ``samples`` contains per-tree
            probability draws.
        """
        preds, samples = self._tree_probabilities(x_test, n_iters, seed)

        middle = self._labels(np.argmax(np.percentile(preds, 50, axis=0), axis=1))
        if n_iters is None:
            return middle
        lower = self._labels(np.argmax(np.percentile(preds, 25, axis=0), axis=1))
        upper = self._labels(np.argmax(np.percentile(preds, 75, axis=0), axis=1))
        return lower, middle, upper, samples

    def predict_proba(self, x_test):
        """Predict class probabilities for a set of test inputs.

        Takes the per-class median of the tree networks' softmax outputs
        (the same reduction :meth:`predict` uses) and normalizes each row.

        Parameters
        ----------
        x_test : array-like of shape (n_test, n_features)
            Input features.

        Returns
        -------
        ndarray
            Probabilities with shape ``(n_test, n_classes)``, columns ordered
            as :attr:`classes_`.
        """
        preds, _ = self._tree_probabilities(x_test, None, None)
        proba = np.median(preds, axis=0).astype(np.float64)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, x_test, seed=None):
        """Predict class labels for a set of test inputs.

        Calls :meth:`bayesian_predict` with ``n_iters=None`` (single
        deterministic forward pass per network) and returns the ``argmax``
        class predictions.

        Parameters
        ----------
        x_test : ndarray
            Input feature matrix for testing.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        ndarray
            Predicted class label for each test point, shape ``(n_test,)``.
        """
        return self.bayesian_predict(x_test, None, seed)


def load(model_path):
    """Load a saved DJINN model from path.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Path to the model directory or its JSON sidecar.

    Returns
    -------
    DJINN_Regressor or DJINN_Classifier
        Reconstructed model with checkpoints loaded.
    """

    path = Path(model_path)
    # find the .json sidecar — could be path itself or path.json
    json_path = path if path.suffix == ".json" else path.with_suffix(".json")
    with open(json_path, "r") as f:
        state = json.load(f)

    # Files from 1.1.x only record the regression flag
    name = state.get("estimator")
    if name is None:
        name = "DJINN_Regressor" if state["regression"] else "DJINN_Classifier"
    cls = DJINN_Classifier if name == "DJINN_Classifier" else DJINN_Regressor

    obj = cls.from_json(json_path)
    obj.load_model(obj.model_name, obj.model_path)
    if cls is DJINN_Classifier and not hasattr(obj, "classes_"):
        obj.classes_ = np.arange(obj.models_[0].output_layer.out_features)
    return obj
