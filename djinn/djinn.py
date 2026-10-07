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
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import MinMaxScaler

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
    device : str or torch.device, optional
        Device used for training and inference.
    """

    _regression = True

    def __init__(
        self,
        n_trees=1,
        max_tree_depth=4,
        dropout_keep_prob=1.0,
        device="cpu",
    ):
        self.n_trees = n_trees
        self.max_tree_depth = max_tree_depth
        self.dropout_keep_prob = dropout_keep_prob
        self.device = device

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
        return {
            "n_trees": self.n_trees_,
            "tree_max_depth": self.max_tree_depth,
            "dropout_keep_prob": self.dropout_keep_prob,
            "regression": self._regression,
            "model_name": model_name,
            "model_path": model_path,
            "xscale": _scaler_state(self.xscale_),
            "yscale": _scaler_state(self.yscale_),
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
            Random seed for reproducibility.

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
        if X.ndim == 1:
            print("Please reshape single-input data to a one-column array")
            return

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
        save_files=True,
        save_model=True,
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
            Random seed for reproducibility.
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

        self.n_trees_ = int(ntrees) if ntrees is not None else self.n_trees
        self.model_name_ = model_name
        self.model_path_ = model_path

        if X.ndim == 1:
            print("Please reshape single-input data to a one-column array")
            return

        single_output = Y.ndim == 1
        if single_output:
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

    def fit(
        self,
        X,
        Y,
        epochs=None,
        learning_rate=None,
        learn_rate=None,
        batch_size=None,
        weight_decay=1.0e-8,
        save_files=True,
        save_model=True,
        model_name="djinn_model",
        model_path="./",
        seed=None,
    ):
        """Train DJINN, auto-selecting hyperparameters when not supplied.

        If ``learning_rate`` is None, calls :meth:`get_hyperparameters` first
        and uses the returned values before delegating to :meth:`train`.

        Parameters
        ----------
        X : ndarray
            Input feature matrix for training.
        Y : ndarray
            Target array for training.
        epochs : int or None, optional
            Number of training epochs.
        learning_rate : float or None, optional
            Learning rate for weight and bias optimization. If ``None``,
            hyperparameters are tuned automatically.
        learn_rate : float or None, optional
            Backward-compatible alias for ``learning_rate``.
        batch_size : int or None, optional
            Number of samples per batch.
        weight_decay : float, optional
            Multiplier for L2 penalty on weights.
        save_files : bool, optional
            If ``True``, saves train/validation cost and weights.
        save_model : bool, optional
            If ``True``, saves the trained model.
        model_name : str, optional
            File name for the model.
        model_path : str, optional
            Directory where model/files are saved.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        self
            The trained model.
        """
        if learn_rate is not None and learning_rate is None:
            learning_rate = learn_rate

        if learning_rate is None:
            optimal = self.get_hyperparameters(X, Y, weight_decay, seed)
            learning_rate = optimal["learning_rate"]
            batch_size = optimal["batch_size"]
            epochs = optimal["epochs"]

        return self.train(
            X=X,
            Y=Y,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            weight_decay=weight_decay,
            save_files=save_files,
            save_model=save_model,
            model_name=model_name,
            model_path=model_path,
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

        if getattr(self, "models_", None) is None:
            self.load_model(self.model_name, self.model_path)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)

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
            return self.yscale_.inverse_transform(raw.cpu().numpy())

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
            ``(n_test, n_outputs)``.
        """
        return self.bayesian_predict(x_test, None, seed)

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
    device : str or torch.device, optional
        Device used for training and inference.
    """


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
    device : str or torch.device, optional
        Device used for training and inference.
    """

    _regression = False

    def bayesian_predict(self, x_test, n_iters, seed=None):
        """Bayesian distribution of class predictions for a set of test inputs.

        Evaluates each tree network ``n_iters`` times (with dropout active)
        to build a predictive distribution over class probabilities, then
        returns the ``argmax`` of the 25th, 50th, and 75th percentiles as
        integer class labels alongside the raw sample dictionary.

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
            indices with shape ``(n_test,)``. Otherwise returns
            ``(lower, middle, upper, samples)``, where percentile outputs are
            1-D arrays of class indices and ``samples`` contains per-tree
            probability draws.
        """

        def to_probabilities(logits):
            # Softmax converts logits to class probabilities
            return torch.softmax(logits, dim=1).cpu().numpy()

        preds, samples = self._tree_outputs(x_test, n_iters, seed, to_probabilities)

        # Reduce probability distributions to class-index predictions
        middle = np.argmax(np.percentile(preds, 50, axis=0), axis=1)
        if n_iters is None:
            return middle
        lower = np.argmax(np.percentile(preds, 25, axis=0), axis=1)
        upper = np.argmax(np.percentile(preds, 75, axis=0), axis=1)
        return lower, middle, upper, samples

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
            Predicted class index for each test point, shape ``(n_test,)``.
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
    DJINN_Regressor
        Reconstructed model with checkpoints loaded.
    """

    path = Path(model_path)
    # find the .json sidecar — could be path itself or path.json
    json_path = path if path.suffix == ".json" else path.with_suffix(".json")
    obj = DJINN_Regressor.from_json(json_path)
    obj.load_model(obj.model_name, obj.model_path)
    return obj
