###############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Written by K. Humbird (humbird1@llnl.gov), L. Peterson (peterson76@llnl.gov).
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
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# Functions from the provided modules
from djinn.neural_network import (
    get_hyperparams,
    load_tree_model,
    torch_continue_training,
    torch_dropout_regression,
)
from djinn.random_forest import fit_scalers, train_forest, tree_to_nn_weights


class DJINN_Regressor:
    """DJINN regression model (PyTorch backend).

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
    **kwargs
        Optional preloaded state including scalers, models, paths, and device.
    """

    def __init__(self, n_trees=1, max_tree_depth=4, dropout_keep_prob=1.0, **kwargs):
        """Initialize a DJINN regressor instance.

        Parameters
        ----------
        n_trees : int, optional
            Number of trees in the random forest (equal to the number of
            neural networks).
        max_tree_depth : int, optional
            Maximum depth of decision tree.
        dropout_keep_prob : float, optional
            Probability of keeping a neuron in dropout layers.
        **kwargs
            Optional preloaded state including ``xscale``, ``yscale``,
            ``regression``, ``models``, ``model_name``, ``model_path``, and
            ``device``.

        Returns
        -------
        None
        """
        self.__n_trees = n_trees
        self.__tree_max_depth = max_tree_depth
        self.__dropout_keep_prob = dropout_keep_prob
        self.__yscale = kwargs.get("yscale", None)
        self.__xscale = kwargs.get("xscale", None)
        self.__regression = kwargs.get("regression", True)
        self.__models = kwargs.get("models", None)
        self.model_name = kwargs.get("model_name", None)
        self.model_path = kwargs.get("model_path", None)
        self.device = torch.device(kwargs.get("device", "cpu"))

    def _fit_scalers(self, X, Y):
        """Fit MinMax scalers on raw data.

        This method is idempotent and only fits scalers when ``self.__xscale``
        is not already set.

        Parameters
        ----------
        X : ndarray
            Raw input feature matrix of shape ``(n_samples, n_features)``.
        Y : ndarray
            Raw target array of shape ``(n_samples,)`` or
            ``(n_samples, n_outputs)``.

        Returns
        -------
        None
        """
        if self.__xscale is None:
            self.__xscale, self.__yscale = fit_scalers(X, Y, self.__regression)

    def _save_json(self):
        """Save model metadata and scalers to a JSON sidecar file.

        Writes ``<model_name>.json`` in ``self.model_path`` for later
        reconstruction via :meth:`from_json`.

        Returns
        -------
        None
        """
        json_path = Path(self.model_path) / f"{self.model_name}.json"
        state = {
            "n_trees": self.__n_trees,
            "tree_max_depth": self.__tree_max_depth,
            "dropout_keep_prob": self.__dropout_keep_prob,
            "regression": self.__regression,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "xscale": {
                "data_min_": self.__xscale.data_min_.tolist(),
                "data_max_": self.__xscale.data_max_.tolist(),
            },
            "yscale": {
                "data_min_": self.__yscale.data_min_.tolist(),
                "data_max_": self.__yscale.data_max_.tolist(),
            },
        }
        with open(json_path, "w") as f:
            json.dump(state, f, indent=2)

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
            Dictionary with keys ``batch_size``, ``learn_rate``, and
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
            self.__n_trees,
            self.__tree_max_depth,
            self.__xscale,
            self.__yscale,
            self.__regression,
            seed,
        )

        tree_to_network = tree_to_nn_weights(
            self.__regression, X, Y, self.__n_trees, rfr, seed
        )

        print("Finding optimal hyper-parameters...")
        nn_batch_size, learnrate, nn_epochs = get_hyperparams(
            self.__regression,
            tree_to_network,
            self.__xscale,
            self.__yscale,
            X,
            Y,
            self.__dropout_keep_prob,
            weight_decay,
            seed=seed,
        )

        return {
            "batch_size": nn_batch_size,
            "learn_rate": learnrate,
            "epochs": nn_epochs,
        }

    def train(
        self,
        X,
        Y,
        epochs=1000,
        learn_rate=0.001,
        batch_size=0,
        weight_decay=1.0e-8,
        save_files=True,
        save_model=True,
        model_name="djinn_model",
        model_path="./",
        seed=None,
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
        learn_rate : float, optional
            Learning rate for weight and bias optimization.
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
        seed : int or None, optional
            Random seed for reproducibility.

        Raises
        ------
        Exception
            If a decision tree cannot be built from the data.

        Returns
        -------
        None
        """
        self.model_name = model_name
        self.model_path = model_path

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
            self.__n_trees,
            self.__tree_max_depth,
            self.__xscale,
            self.__yscale,
            self.__regression,
            seed,
        )

        tree_to_network = tree_to_nn_weights(
            self.__regression, X, Y, self.__n_trees, rfr, seed
        )

        if batch_size == 0:
            batch_size = int(np.ceil(0.05 * len(Y)))

        self.nninfo = torch_dropout_regression(
            self.__regression,
            tree_to_network,
            self.__xscale,
            self.__yscale,
            X,
            Y,
            ntrees=self.__n_trees,
            lr=learn_rate,
            n_epochs=epochs,
            batch_size=batch_size,
            dropout_keep_prob=self.__dropout_keep_prob,
            weight_decay=weight_decay,
            # kwargs forwarded to torch_dropout_regression
            save_model=save_model,
            save_files=save_files,
            model_path=str(Path(model_path) / model_name),
            seed=seed,
            device=self.device,
        )

        if save_model:
            self._save_json()

    def fit(
        self,
        X,
        Y,
        epochs=None,
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

        If ``learn_rate`` is None, calls :meth:`get_hyperparameters` first
        and uses the returned values before delegating to :meth:`train`.

        Parameters
        ----------
        X : ndarray
            Input feature matrix for training.
        Y : ndarray
            Target array for training.
        epochs : int or None, optional
            Number of training epochs.
        learn_rate : float or None, optional
            Learning rate for weight and bias optimization. If ``None``,
            hyperparameters are tuned automatically.
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
        None
        """
        if learn_rate is None:
            optimal = self.get_hyperparameters(X, Y, weight_decay, seed)
            learn_rate = optimal["learn_rate"]
            batch_size = optimal["batch_size"]
            epochs = optimal["epochs"]

        self.train(
            X=X,
            Y=Y,
            epochs=epochs,
            learn_rate=learn_rate,
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
        """Reconstruct a DJINN_Regressor from a saved JSON state file.

        Restores all hyperparameters and scalers so the instance is ready
        for :meth:`load_model`, :meth:`predict`, or :meth:`continue_training`.

        Parameters
        ----------
        json_path : str or pathlib.Path
            Path to the ``.json`` file written by :meth:`train`.

        Returns
        -------
        DJINN_Regressor
            Restored regressor instance.
        """
        with open(json_path, "r") as f:
            state = json.load(f)

        obj = cls(
            n_trees=state["n_trees"],
            max_tree_depth=state["tree_max_depth"],
            dropout_keep_prob=state["dropout_keep_prob"],
        )
        obj._DJINN_Regressor__regression = state["regression"]
        obj.model_name = state["model_name"]
        obj.model_path = state["model_path"]

        xscale = MinMaxScaler()
        xscale.data_min_ = np.array(state["xscale"]["data_min_"])
        xscale.data_max_ = np.array(state["xscale"]["data_max_"])
        xscale.scale_ = xscale.data_max_ - xscale.data_min_
        xscale.data_range_ = xscale.scale_.copy()
        xscale.min_ = -xscale.data_min_ / xscale.scale_
        xscale.n_features_in_ = xscale.data_min_.shape[0]
        obj._DJINN_Regressor__xscale = xscale

        yscale = MinMaxScaler()
        yscale.data_min_ = np.array(state["yscale"]["data_min_"])
        yscale.data_max_ = np.array(state["yscale"]["data_max_"])
        yscale.scale_ = yscale.data_max_ - yscale.data_min_
        yscale.data_range_ = yscale.scale_.copy()
        yscale.min_ = -yscale.data_min_ / yscale.scale_
        yscale.n_features_in_ = yscale.data_min_.shape[0]
        obj._DJINN_Regressor__yscale = yscale

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

        self.__models = {}
        for tree_idx in range(self.__n_trees):
            checkpoint_path = model_dir / f"tree_{tree_idx}.pt"
            model, _ = load_tree_model(
                checkpoint_path, self.device, self.__dropout_keep_prob, tree_idx
            )
            self.__models[tree_idx] = model

    def close_model(self):
        """Release all loaded PyTorch models from memory.

        Returns
        -------
        None
        """
        self.__models = None

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
        non_bayes = n_iters is None
        if non_bayes:
            n_iters = 1

        if seed is not None:
            torch.manual_seed(seed)

        if self.__models is None:
            self.load_model(self.model_name, self.model_path)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)

        samples = {"inputs": x_test, "predictions": {}}

        self.__xscale.clip = False
        x_scaled = self.__xscale.transform(x_test)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)

        for tree_idx in range(self.__n_trees):
            model = self.__models[tree_idx].to(self.device)
            model.train()  # keep dropout active for Bayesian sampling

            tree_preds = []
            with torch.no_grad():
                for _ in range(n_iters):
                    raw = model(x_tensor).cpu().numpy()
                    pred = self.__yscale.inverse_transform(raw)
                    tree_preds.append(pred)

            samples["predictions"][f"tree{tree_idx}"] = tree_preds

        n_out = samples["predictions"]["tree0"][0].shape[1]
        preds = np.array(
            [samples["predictions"][t] for t in samples["predictions"]]
        ).reshape((n_iters * self.__n_trees, len(x_test), n_out))

        middle = np.percentile(preds, 50, axis=0)
        lower = np.percentile(preds, 25, axis=0)
        upper = np.percentile(preds, 75, axis=0)

        if non_bayes:
            return np.mean(preds, axis=0)
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
            (n_iters * self.__n_trees, x_length, n_out)
        )
        return preds

    def continue_training(
        self,
        X,
        Y,
        training_epochs,
        learn_rate,
        batch_size,
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
        learn_rate : float
            Learning rate.
        batch_size : int
            Number of samples per batch.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        None
        """
        model_dir = Path(self.model_path) / self.model_name

        torch_continue_training(
            regression=self.__regression,
            xscale=self.__xscale,
            yscale=self.__yscale,
            x=X,
            y=Y,
            ntrees=self.__n_trees,
            lr=learn_rate,
            n_epochs=training_epochs,
            batch_size=batch_size,
            dropout_keep_prob=self.__dropout_keep_prob,
            model_dir=model_dir,
            model_name=self.model_name,
            weight_decay=0.0,
            seed=seed,
            device=self.device,
        )


class DJINN_Classifier(DJINN_Regressor):
    """DJINN classification model.

    Inherits all training, saving, and loading behaviour from
    :class:`DJINN_Regressor`. The only behavioural difference is in
    :meth:`bayesian_predict`, where no output scaling is applied and
    ``np.argmax`` is used to convert softmax distributions into class
    predictions.

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
    **kwargs
        Optional keyword arguments forwarded to :class:`DJINN_Regressor`.
    """

    def __init__(self, n_trees=1, max_tree_depth=4, dropout_keep_prob=1.0, **kwargs):
        """Initialize a DJINN classifier instance.

        Parameters
        ----------
        n_trees : int, optional
            Number of trees in the random forest (equal to the number of
            neural networks).
        max_tree_depth : int, optional
            Maximum depth of decision tree.
        dropout_keep_prob : float, optional
            Probability of keeping a neuron in dropout layers.
        **kwargs
            Optional keyword arguments forwarded to
            :class:`DJINN_Regressor`.

        Returns
        -------
        None
        """
        super().__init__(n_trees, max_tree_depth, dropout_keep_prob, **kwargs)
        # Override the regression flag set by the parent
        self._DJINN_Regressor__regression = False

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
        non_bayes = n_iters is None
        if non_bayes:
            n_iters = 1

        if seed is not None:
            torch.manual_seed(seed)

        if self._DJINN_Regressor__models is None:
            self.load_model(self.model_name, self.model_path)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)

        samples = {"inputs": x_test, "predictions": {}}

        self._DJINN_Regressor__xscale.clip = False
        x_scaled = self._DJINN_Regressor__xscale.transform(x_test)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)

        n_trees = self._DJINN_Regressor__n_trees
        # dropout_keep_prob = self._DJINN_Regressor__dropout_keep_prob

        for tree_idx in range(n_trees):
            model = self._DJINN_Regressor__models[tree_idx].to(self.device)
            model.train()  # keep dropout active for Bayesian sampling

            tree_preds = []
            with torch.no_grad():
                for _ in range(n_iters):
                    # Softmax converts logits to class probabilities
                    logits = model(x_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    tree_preds.append(probs)

            samples["predictions"][f"tree{tree_idx}"] = tree_preds

        n_out = samples["predictions"]["tree0"][0].shape[1]
        preds = np.array(
            [samples["predictions"][t] for t in samples["predictions"]]
        ).reshape((n_iters * n_trees, len(x_test), n_out))

        # Reduce probability distributions to class-index predictions
        middle = np.argmax(np.percentile(preds, 50, axis=0), axis=1)
        lower = np.argmax(np.percentile(preds, 25, axis=0), axis=1)
        upper = np.argmax(np.percentile(preds, 75, axis=0), axis=1)

        if non_bayes:
            return middle
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
