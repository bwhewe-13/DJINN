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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def scale_data(x, y, xscale, yscale, regression, seed, n_classes):
    """Scale data and split into a training subset.

    Parameters
    ----------
    x : ndarray
        Input feature matrix.
    y : ndarray
        Output target array.
    xscale : object
        Fitted scaler used to transform ``x``.
    yscale : object
        Fitted scaler used to transform ``y`` for regression.
    regression : bool
        Whether the task is regression or classification.
    seed : int
        Random seed used for ``train_test_split``.
    n_classes : int
        Number of output classes/targets.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Scaled ``(xtrain, ytrain)`` tensors.
    """
    # Copy DataFrames to avoid modifying original data
    x = x.copy()
    y = y.copy()

    x = xscale.transform(x)
    if regression:
        if n_classes == 1:
            y = yscale.transform(y).flatten()
        else:
            y = yscale.transform(y)

    xtrain, _, ytrain, _ = train_test_split(x, y, test_size=0.1, random_state=seed)

    # for classification, do one-hot encoding on classes
    if not regression:
        ytrain = F.one_hot(ytrain.flatten(), num_classes=torch.unique(ytrain).numel())

    x_tensor = torch.tensor(xtrain, dtype=torch.float32)
    y_tensor = torch.tensor(ytrain, dtype=torch.float32)

    return x_tensor, y_tensor


class MultiLayerPerceptron(nn.Module):
    """Feed-forward network initialized from DJINN tree-mapped weights."""

    def __init__(self, weights, biases, dropout_keep_prob):
        """Initialize the network.

        Parameters
        ----------
        weights : dict
            Layer weight matrices keyed by ``h1..hN`` and ``out``.
        biases : dict
            Layer bias vectors keyed by ``h1..hN`` and ``out``.
        dropout_keep_prob : float
            Probability of keeping hidden activations.
        """
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=1.0 - dropout_keep_prob)

        # Number of hidden layers inferred from weight keys
        nhl = len([k for k in weights.keys() if k.startswith("h")])

        # Hidden Layers
        for i in range(1, nhl + 1):
            w = torch.as_tensor(weights[f"h{i}"], dtype=torch.float32)
            b = torch.as_tensor(biases[f"h{i}"], dtype=torch.float32)

            # NOTE: TensorFlow Dense kernel shape = (in, out)
            #       PyTorch Linear weight shape = (out, in)
            layer = nn.Linear(w.shape[0], w.shape[1])

            with torch.no_grad():
                layer.weight.copy_(w.T)
                layer.bias.copy_(b)

            self.hidden_layers.append(layer)

        # Output Layer
        w_out = torch.as_tensor(weights["out"], dtype=torch.float32)
        b_out = torch.as_tensor(biases["out"], dtype=torch.float32)

        self.output_layer = nn.Linear(w_out.shape[0], w_out.shape[1])

        with torch.no_grad():
            self.output_layer.weight.copy_(w_out.T)
            self.output_layer.bias.copy_(b_out)

    def forward(self, x):
        """Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input minibatch tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after hidden and output layers.
        """
        for i, layer in enumerate(self.hidden_layers):
            x = F.relu(layer(x))
            if i >= 1:
                x = self.dropout(x)
        x = self.output_layer(x)
        return x


def _l2_kernel_penalty(model, weight_decay):
    """Compute L2 penalty over kernel weights only.

    Parameters
    ----------
    model : MultiLayerPerceptron
        Network containing hidden and output linear layers.
    weight_decay : float
        L2 multiplier.

    Returns
    -------
    float or torch.Tensor
        Zero when ``weight_decay`` is falsy, otherwise the scaled L2 penalty term.
    """
    if not weight_decay:
        return 0.0

    penalty = 0.0
    for layer in model.hidden_layers:
        penalty = penalty + torch.sum(layer.weight.pow(2))
    penalty = penalty + torch.sum(model.output_layer.weight.pow(2))
    return 0.5 * weight_decay * penalty


def build_tree_weights_and_biases(ttn, key):
    """Construct per-layer weights and random biases from mapped tree data.

    Parameters
    ----------
    ttn : dict
        Output from tree-to-network mapping.
    key : str
        Tree key (for example ``tree_0``).

    Returns
    -------
    tuple[dict, dict]
        ``(weights, biases)`` dictionaries keyed by ``h1..hN`` and ``out``.
    """
    npl = ttn["network_shape"][key]
    nhl = len(npl) - 2
    n_classes = ttn["n_out"]

    # Transposed weights from DJINN
    w = {
        i + 1: np.transpose(ttn["weights"][key][i]).astype(np.float32)
        for i in range(len(npl) - 1)
    }

    weights = {f"h{i}": w[i] for i in range(1, nhl + 1)}
    weights["out"] = w[nhl + 1]

    # Random biases (same formula as original)
    n_hidden_last = npl[-2]
    scale = np.sqrt(3.0 / (n_classes + n_hidden_last))

    biases = {
        f"h{i}": np.random.normal(0.0, scale, size=(npl[i],)).astype(np.float32)
        for i in range(1, nhl + 1)
    }

    biases["out"] = np.random.normal(0.0, scale, size=(n_classes,)).astype(np.float32)
    return weights, biases


def prepare_dataloader(xtrain, ytrain, regression, batch_size, device):
    """Create a shuffled PyTorch dataloader for regression or classification.

    Parameters
    ----------
    xtrain : ndarray or torch.Tensor
        Training features.
    ytrain : ndarray or torch.Tensor
        Training targets.
    regression : bool
        True for regression, False for classification.
    batch_size : int
        Mini-batch size.
    device : torch.device
        Device where tensors are materialized.

    Returns
    -------
    torch.utils.data.DataLoader
        Shuffled training dataloader.
    """

    xtrain = torch.tensor(xtrain, dtype=torch.float32, device=device)

    if regression:
        ytrain = torch.tensor(ytrain, dtype=torch.float32, device=device)
    else:
        if ytrain.ndim > 1:
            ytrain = torch.argmax(torch.tensor(ytrain, dtype=torch.float32), dim=1)
        else:
            ytrain = torch.tensor(ytrain, dtype=torch.long)
        ytrain = ytrain.to(device)

    dataset = TensorDataset(xtrain, ytrain)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(
    model,
    loader,
    regression,
    lr,
    weight_decay,
    epochs=100,
    early_stop_patience=None,
):
    """Train a model and return epoch losses.

    Parameters
    ----------
    model : nn.Module
        Model to optimize.
    loader : torch.utils.data.DataLoader
        Training batches.
    regression : bool
        True for regression (MSE), False for classification.
    lr : float
        Adam learning rate.
    weight_decay : float
        L2 regularization strength for kernel weights.
    epochs : int, optional
        Maximum number of epochs.
    early_stop_patience : int or None, optional
        Optional early-stop patience.

    Returns
    -------
    list[float]
        Mean training loss per epoch.
    """

    if regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            if weight_decay:
                loss = loss + _l2_kernel_penalty(model, weight_decay)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        losses.append(epoch_loss)

        # Optional early stopping
        if early_stop_patience:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break

    return losses


def get_learning_rate(
    regression,
    weights,
    biases,
    xtrain,
    ytrain,
    dropout_keep_prob,
    batch_size,
    weight_decay=0.0,
    device=None,
):
    """Search for an effective learning rate using two refinement passes.

    Parameters
    ----------
    regression : bool
        True for regression, False for classification.
    weights : dict
        Initial per-layer weights.
    biases : dict
        Initial per-layer biases.
    xtrain : ndarray or torch.Tensor
        Training features.
    ytrain : ndarray or torch.Tensor
        Training labels/targets.
    dropout_keep_prob : float
        Hidden-layer dropout keep probability.
    batch_size : int
        Mini-batch size.
    weight_decay : float, optional
        L2 regularization strength.
    device : torch.device or None, optional
        Compute device.

    Returns
    -------
    float
        Selected learning rate.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = prepare_dataloader(xtrain, ytrain, regression, batch_size, device)

    minlr = -4.0
    maxlr = -2.0
    lrs = np.logspace(minlr, maxlr, 10)

    # two refinement passes
    for _ in range(2):

        errormin = []
        for lr in lrs:

            model = MultiLayerPerceptron(weights, biases, dropout_keep_prob).to(device)

            losses = train_model(
                model,
                loader,
                regression,
                lr,
                weight_decay,
                epochs=100,
                early_stop_patience=None,
            )

            errormin.append(np.mean(losses[90:]))

        errormin = np.array(errormin)
        indices = errormin.argsort()[:2]

        minlr = min(lrs[indices[0]], lrs[indices[1]])
        maxlr = max(lrs[indices[0]], lrs[indices[1]])

        lrs = np.linspace(minlr, maxlr, 10)

    return minlr


def find_optimal_epochs(
    regression,
    weights,
    biases,
    xtrain,
    ytrain,
    dropout_keep_prob,
    lr,
    batch_size,
    weight_decay=0.0,
    max_training_epochs=3000,
    device=None,
):
    """Estimate epochs-to-convergence for fixed hyperparameters.

    Parameters
    ----------
    regression : bool
        True for regression, False for classification.
    weights : dict
        Initial per-layer weights.
    biases : dict
        Initial per-layer biases.
    xtrain : ndarray or torch.Tensor
        Training features.
    ytrain : ndarray or torch.Tensor
        Training labels/targets.
    dropout_keep_prob : float
        Hidden-layer dropout keep probability.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    weight_decay : float, optional
        L2 regularization strength.
    max_training_epochs : int, optional
        Upper bound on epochs.
    device : torch.device or None, optional
        Compute device.

    Returns
    -------
    int
        Estimated number of training epochs.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataloader once
    loader = prepare_dataloader(xtrain, ytrain, regression, batch_size, device)

    # Build model once (same behavior as TF version)
    model = MultiLayerPerceptron(weights, biases, dropout_keep_prob).to(device)

    # Loss
    if regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)

    accur = []
    epoch = 0
    converged = False

    epoch = 200
    for _ in range(epoch):
        epoch_loss = _train_one_epoch(model, loader, criterion, optimizer, weight_decay)
        accur.append(epoch_loss)

    while not converged and epoch < max_training_epochs:

        for _ in range(10):
            epoch_loss = _train_one_epoch(
                model, loader, criterion, optimizer, weight_decay
            )
            accur.append(epoch_loss)

        epoch += 10

        if epoch >= 30:
            upper = np.mean(accur[epoch - 10 : epoch])
            middle = np.mean(accur[epoch - 20 : epoch - 10])
            lower = np.mean(accur[epoch - 30 : epoch - 20])

            d1 = 100 * abs(upper - middle) / (upper + 1e-8)
            d2 = 100 * abs(middle - lower) / (middle + 1e-8)

            if d1 < 5 and d2 < 5:
                converged = True
                maxep = epoch

        if epoch >= max_training_epochs:
            converged = True
            print("Warning: Reached max # training epochs:", max_training_epochs)
            maxep = max_training_epochs

    return maxep


def _train_one_epoch(model, loader, criterion, optimizer, weight_decay=0.0):
    """Train a single epoch and return mean batch loss.

    Parameters
    ----------
    model : nn.Module
        Model to optimize.
    loader : torch.utils.data.DataLoader
        Training batches.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer used for parameter updates.
    weight_decay : float, optional
        L2 regularization strength.

    Returns
    -------
    float
        Mean loss across batches for the epoch.
    """
    model.train()
    running_loss = 0.0

    for xb, yb in loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        if weight_decay:
            loss = loss + _l2_kernel_penalty(model, weight_decay)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def get_hyperparams(
    regression,
    ttn,
    xscale,
    yscale,
    x,
    y,
    dropout_keep_prob,
    weight_decay,
    seed,
    device=None,
):
    """Automatically select DJINN hyperparameters for a mapped tree network.

    Parameters
    ----------
    regression : bool
        Whether the task is regression or classification.
    ttn : dict
        Dictionary returned by tree-to-network mapping.
    xscale : object
        Input scaler.
    yscale : object
        Output scaler.
    x : ndarray
        Input features.
    y : ndarray
        Output features.
    dropout_keep_prob : float
        Probability of keeping a hidden unit active during dropout.
    weight_decay : float
        Multiplier for L2 penalty on weights.
    seed : int
        Random seed used in preprocessing/splitting.
    device : torch.device or None, optional
        Compute device.

    Returns
    -------
    tuple[int, float, int]
        ``(batch_size, learning_rate, epochs)``.
    """

    # Scale data
    batch_size = int(np.ceil(0.05 * len(y)))
    xtrain, ytrain = scale_data(x, y, xscale, yscale, regression, seed, ttn["n_out"])

    ystar = {}
    ystar["preds"] = {}

    print("Determining learning rate...")
    key = "tree_0"
    weights, biases = build_tree_weights_and_biases(ttn, key)
    lr = get_learning_rate(
        regression,
        weights,
        biases,
        xtrain,
        ytrain,
        dropout_keep_prob,
        batch_size,
        weight_decay,
        device,
    )

    print("Determining number of epochs needed...")
    max_training_epochs = 3000
    opt_epochs = find_optimal_epochs(
        regression,
        weights,
        biases,
        xtrain,
        ytrain,
        dropout_keep_prob,
        lr,
        batch_size,
        weight_decay,
        max_training_epochs,
        device,
    )

    print("Optimal learning rate: ", lr)
    print("Optimal # epochs: ", opt_epochs)
    print("Optimal batch size: ", batch_size)

    return (batch_size, lr, opt_epochs)
