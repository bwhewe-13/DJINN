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

"""PyTorch training, evaluation, and persistence utilities for DJINN models."""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that serializes NumPy arrays as Python lists."""

    def default(self, obj):
        """Convert NumPy arrays to lists; delegate all other types to super.

        Parameters
        ----------
        obj : object
            The object to serialize.

        Returns
        -------
        list or default
            A Python list if ``obj`` is an ``ndarray``; otherwise the result
            of the parent encoder's ``default`` method.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def scale_data(x, y, xscale, yscale, regression, seed, n_classes, test=False):
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

    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.1, random_state=seed
    )

    # for classification, do one-hot encoding on classes
    if not regression:
        ytrain = F.one_hot(
            torch.as_tensor(ytrain.flatten(), dtype=torch.long), num_classes=n_classes
        )
        ytest = F.one_hot(
            torch.as_tensor(ytest.flatten(), dtype=torch.long), num_classes=n_classes
        )

    xtrain_tensor = torch.as_tensor(xtrain, dtype=torch.float32)
    xtest_tensor = torch.as_tensor(xtest, dtype=torch.float32)
    ytrain_tensor = torch.as_tensor(ytrain, dtype=torch.float32)
    ytest_tensor = torch.as_tensor(ytest, dtype=torch.float32)

    if test:
        return xtrain_tensor, xtest_tensor, ytrain_tensor, ytest_tensor
    return xtrain_tensor, ytrain_tensor


class MultiLayerPerceptron(nn.Module):
    """Feed-forward network initialized from DJINN tree-mapped weights.

    Notes
    -----
    Hidden layers and output layer are rebuilt from per-layer weight and bias
    dictionaries produced by DJINN tree-to-network mapping.
    """

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


def build_tree_weights_and_biases(ttn, key, seed=None, tree_idx=0):
    """Construct per-layer weights and random biases from mapped tree data.

    Parameters
    ----------
    ttn : dict
        Output from tree-to-network mapping.
    key : str
        Tree key (for example ``tree_0``).
    seed : int or None, optional
        Base random seed. Combined with ``tree_idx`` so each tree in the
        ensemble gets a distinct but reproducible bias initialisation,
        regardless of how much numpy random state was consumed beforehand.
    tree_idx : int, optional
        Zero-based index of the tree, used to derive a per-tree seed.

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

    # Seed numpy immediately before bias draws so the initialisation is
    # reproducible regardless of how much random state was consumed by
    # earlier operations (e.g. train_test_split inside scale_data).
    # Each tree gets a unique seed derived from the base seed so the
    # ensemble members are not identically initialised.
    rng = np.random.default_rng(None if seed is None else seed * 1000 + tree_idx)

    # Random biases (same formula as original)
    n_hidden_last = npl[-2]
    scale = np.sqrt(3.0 / (n_classes + n_hidden_last))

    biases = {
        f"h{i}": rng.standard_normal(size=(npl[i],)).astype(np.float32) * scale
        for i in range(1, nhl + 1)
    }

    biases["out"] = rng.standard_normal(size=(n_classes,)).astype(np.float32) * scale
    return weights, biases


def prepare_dataloader(xtrain, ytrain, regression, batch_size, device, seed=None):
    """Create a minibatch dataloader for regression or classification.

    Matches the original TensorFlow DJINN: each epoch draws
    ``len(xtrain) // batch_size`` full batches of rows sampled with
    replacement, rather than a shuffled pass over every row. On the diabetes
    benchmark this is worth about +0.02 R² over a shuffled pass.

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
    seed : int or None, optional
        Seed for the sampling generator.

    Returns
    -------
    BatchLoader
        Training dataloader.
    """

    xtrain = torch.as_tensor(xtrain, dtype=torch.float32, device=device)

    if regression:
        ytrain = torch.as_tensor(ytrain, dtype=torch.float32, device=device)
        if ytrain.ndim == 1:
            ytrain = ytrain.unsqueeze(1)
    else:
        if ytrain.ndim > 1:
            ytrain = torch.argmax(torch.as_tensor(ytrain, dtype=torch.float32), dim=1)
        else:
            ytrain = torch.as_tensor(ytrain, dtype=torch.long)
        ytrain = ytrain.to(device)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return BatchLoader(xtrain, ytrain, batch_size, generator)


class BatchLoader:
    """Minibatches of rows drawn with replacement.

    Gives the same batches as a ``DataLoader`` with a ``RandomSampler``
    (``replacement=True``), but indexes the tensors directly instead of
    collating one row at a time, which was most of the training time.
    """

    def __init__(self, x, y, batch_size, generator):
        """Store the data and the sampling generator.

        Parameters
        ----------
        x : torch.Tensor
            Training features.
        y : torch.Tensor
            Training targets.
        batch_size : int
            Mini-batch size.
        generator : torch.Generator
            Generator used to draw the rows.
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.generator = generator
        self.n_batches = max(1, len(x) // batch_size)

    def __len__(self):
        """Return the number of batches per epoch."""
        return self.n_batches

    def __iter__(self):
        """Yield ``(x, y)`` batches for one epoch."""
        n_samples = self.n_batches * self.batch_size
        idx = draw_rows(len(self.x), n_samples, self.generator).to(self.x.device)
        for batch in idx.split(self.batch_size):
            yield self.x[batch], self.y[batch]


def draw_rows(n, n_samples, generator):
    """Draw ``n_samples`` row indices from ``range(n)`` with replacement.

    Uses the same calls as ``RandomSampler(replacement=True)``, in chunks of
    32, so a seeded generator gives the same rows as before.
    """
    chunks = [
        torch.randint(n, (32,), dtype=torch.int64, generator=generator)
        for _ in range(n_samples // 32)
    ]
    chunks.append(
        torch.randint(n, (n_samples % 32,), dtype=torch.int64, generator=generator)
    )
    return torch.cat(chunks)


def stack_networks(weights_list, biases_list, device):
    """Zero-pad networks with the same number of layers and stack them.

    The padded neurons have no inputs, outputs or bias, so their gradients
    are zero and they stay zero during training.

    Parameters
    ----------
    weights_list, biases_list : list[dict]
        Per-network weights and biases from
        :func:`build_tree_weights_and_biases`.
    device : torch.device
        Device for the stacked tensors.

    Returns
    -------
    tuple[list[str], list[torch.Tensor], list[torch.Tensor]]
        Layer keys, weights of shape ``(n_nets, n_in, n_out)`` and biases of
        shape ``(n_nets, 1, n_out)``.
    """
    n_hidden = len([k for k in weights_list[0] if k.startswith("h")])
    keys = [f"h{i}" for i in range(1, n_hidden + 1)] + ["out"]
    n_nets = len(weights_list)
    stacked_w, stacked_b = [], []
    for key in keys:
        n_in = max(w[key].shape[0] for w in weights_list)
        n_out = max(w[key].shape[1] for w in weights_list)
        w_stack = torch.zeros(n_nets, n_in, n_out)
        b_stack = torch.zeros(n_nets, 1, n_out)
        for i, (w, b) in enumerate(zip(weights_list, biases_list)):
            rows, cols = w[key].shape
            w_stack[i, :rows, :cols] = torch.as_tensor(w[key], dtype=torch.float32)
            b_stack[i, 0, :cols] = torch.as_tensor(b[key], dtype=torch.float32)
        stacked_w.append(w_stack.to(device).requires_grad_())
        stacked_b.append(b_stack.to(device).requires_grad_())
    return keys, stacked_w, stacked_b


def unstack_network(keys, stacked_w, stacked_b, idx, shapes):
    """Return network ``idx`` from a stack as weight and bias dicts."""
    weights, biases = {}, {}
    for key, w, b in zip(keys, stacked_w, stacked_b):
        rows, cols = shapes[key]
        weights[key] = w[idx, :rows, :cols].detach().cpu().numpy().copy()
        biases[key] = b[idx, 0, :cols].detach().cpu().numpy().copy()
    return weights, biases


def stacked_forward(stacked_w, stacked_b, x, dropout_p=0.0):
    """Forward pass of stacked networks, like :class:`MultiLayerPerceptron`.

    ``x`` has shape ``(n_nets, n_rows, n_features)``. Dropout is applied
    after the second and later hidden layers when ``dropout_p > 0``.
    """
    for i, (w, b) in enumerate(zip(stacked_w[:-1], stacked_b[:-1])):
        x = torch.relu(torch.baddbmm(b, x, w))
        if i >= 1 and dropout_p > 0:
            x = F.dropout(x, dropout_p, training=True)
    return torch.baddbmm(stacked_b[-1], x, stacked_w[-1])


def stacked_loss(outputs, targets, regression):
    """Mean loss of each network, shape ``(n_nets,)``.

    Targets are floats of shape ``(n_nets, n_rows, n_out)`` for regression
    and class indices of shape ``(n_nets, n_rows)`` for classification.
    """
    if regression:
        return ((outputs - targets) ** 2).mean(dim=(1, 2))
    n_nets, n_rows, n_out = outputs.shape
    loss = F.cross_entropy(
        outputs.reshape(-1, n_out), targets.reshape(-1), reduction="none"
    )
    return loss.reshape(n_nets, n_rows).mean(dim=1)


class StackedAdam:
    """Adam for stacked networks, with a learning rate per network.

    Follows ``torch.optim.Adam`` (L2 ``weight_decay`` added to the gradient).
    Each network's update only depends on its own slice, so this is the same
    as running a separate Adam for every network.
    """

    def __init__(self, params, lr, weight_decay=0.0, eps=1e-8, betas=(0.9, 0.999)):
        """Set up the optimizer state.

        Parameters
        ----------
        params : list[torch.Tensor]
            Stacked parameters, with the network as the first dimension.
        lr : float or torch.Tensor
            Learning rate, or one per network with shape ``(n_nets, 1, 1)``.
        weight_decay : float, optional
            L2 penalty added to the gradient.
        eps : float, optional
            Term added to the denominator.
        betas : tuple[float, float], optional
            Decay rates of the moment estimates.
        """
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.betas = betas
        self.step_count = 0
        self.exp_avg = [torch.zeros_like(p) for p in params]
        self.exp_avg_sq = [torch.zeros_like(p) for p in params]

    def zero_grad(self):
        """Clear the gradients."""
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        """Update every parameter once."""
        beta1, beta2 = self.betas
        self.step_count += 1
        bias_correction1 = 1 - beta1**self.step_count
        bias_correction2_sqrt = (1 - beta2**self.step_count) ** 0.5
        step_size = self.lr / bias_correction1
        for p, m, v in zip(self.params, self.exp_avg, self.exp_avg_sq):
            grad = p.grad
            if self.weight_decay:
                grad = grad.add(p, alpha=self.weight_decay)
            m.lerp_(grad, 1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = (v.sqrt() / bias_correction2_sqrt).add_(self.eps)
            p.sub_(step_size * m / denom)


def train_stacked(
    stacked_w,
    stacked_b,
    xtrain,
    ytrain,
    regression,
    generators,
    batch_size,
    epochs,
    optimizer,
    dropout_p=0.0,
    xval=None,
    yval=None,
    eval_every=1,
):
    """Train stacked networks together, each on its own minibatches.

    Parameters
    ----------
    stacked_w, stacked_b : list[torch.Tensor]
        Parameters from :func:`stack_networks`, updated in place.
    xtrain : torch.Tensor
        Training features, shape ``(n_rows, n_features)``.
    ytrain : torch.Tensor
        Float targets of shape ``(n_rows, n_out)`` for regression, class
        indices of shape ``(n_rows,)`` for classification.
    regression : bool
        True for regression (MSE), False for classification.
    generators : list[torch.Generator]
        One generator per network for drawing its minibatches.
    batch_size : int
        Mini-batch size.
    epochs : int
        Number of epochs.
    optimizer : StackedAdam
        Optimizer for the stacked parameters.
    dropout_p : float, optional
        Dropout probability.
    xval, yval : torch.Tensor or None, optional
        Validation data, same layout as the training data.
    eval_every : int, optional
        Evaluate the validation set every this many epochs.

    Returns
    -------
    tuple[ndarray, ndarray or None]
        Training and validation loss, shape ``(n_nets, epochs)``.
    """
    n_nets = len(generators)
    n_rows = len(xtrain)
    n_batches = max(1, n_rows // batch_size)
    n_samples = n_batches * batch_size
    device = xtrain.device
    if xval is not None:
        xval_stack = xval.expand(n_nets, *xval.shape)
        yval_stack = yval.expand(n_nets, *yval.shape)
        val_loss = torch.full((n_nets,), float("nan"), device=device)

    train_history, valid_history = [], []
    for epoch in range(epochs):
        rows = torch.stack([draw_rows(n_rows, n_samples, g) for g in generators])
        rows = rows.to(device).view(n_nets, n_batches, batch_size)
        total = torch.zeros(n_nets, device=device)
        for k in range(n_batches):
            batch = rows[:, k]
            outputs = stacked_forward(stacked_w, stacked_b, xtrain[batch], dropout_p)
            loss = stacked_loss(outputs, ytrain[batch], regression)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            total += loss.detach()
        train_history.append(total / n_batches)

        if xval is not None:
            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    outputs = stacked_forward(stacked_w, stacked_b, xval_stack)
                    val_loss = stacked_loss(outputs, yval_stack, regression)
            valid_history.append(val_loss)

    train_history = torch.stack(train_history, dim=1).cpu().numpy()
    if xval is None:
        return train_history, None
    return train_history, torch.stack(valid_history, dim=1).cpu().numpy()


def train_one_epoch(model, loader, criterion, optimizer):
    """Run one optimization epoch over the training loader.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    loader : BatchLoader
        Mini-batch dataloader for training data.
    criterion : callable
        Loss function used to compute training loss.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.

    Returns
    -------
    float
        Mean loss across all mini-batches in the epoch.
    """
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()

    return float(total_loss) / len(loader)


@torch.no_grad()
def evaluate(model, x, y, criterion):
    """Evaluate a model on a fixed dataset without gradient tracking.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    x : torch.Tensor
        Input features for evaluation.
    y : torch.Tensor
        Targets corresponding to ``x``.
    criterion : callable
        Loss function used to score predictions.

    Returns
    -------
    float
        Scalar loss value on the provided dataset.
    """
    model.eval()
    outputs = model(x)
    return criterion(outputs, y).item()


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
    loader : BatchLoader
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

    criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )

    losses = []
    best_loss = float("inf")
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        epoch_loss = train_one_epoch(model, loader, criterion, optimizer)
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
    device : torch.device, str, or None, optional
        Compute device.

    Returns
    -------
    float
        Selected learning rate.
    """

    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = prepare_dataloader(xtrain, ytrain, regression, batch_size, device)
    n_rows = len(loader.x)
    n_samples = loader.n_batches * batch_size
    epochs = 100

    minlr = -4.0
    maxlr = -2.0
    lrs = np.logspace(minlr, maxlr, 10)

    # two refinement passes, training all the learning rates together
    for _ in range(2):
        # Give each learning rate the minibatches it would get if they were
        # trained one after another from the same generator
        generators = []
        for _ in lrs:
            generator = torch.Generator()
            generator.set_state(loader.generator.get_state())
            generators.append(generator)
            for _ in range(epochs):
                draw_rows(n_rows, n_samples, loader.generator)

        layer_keys, stacked_w, stacked_b = stack_networks(
            [weights] * len(lrs), [biases] * len(lrs), device
        )
        lr_tensor = torch.tensor(lrs, dtype=torch.float32, device=device)
        optimizer = StackedAdam(
            stacked_w + stacked_b,
            lr_tensor.view(-1, 1, 1),
            weight_decay,
            eps=1e-8,
        )
        losses, _ = train_stacked(
            stacked_w,
            stacked_b,
            loader.x,
            loader.y,
            regression,
            generators,
            batch_size,
            epochs,
            optimizer,
            dropout_p=1.0 - dropout_keep_prob,
        )
        errormin = losses[:, 90:].mean(axis=1)

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
    device : torch.device, str, or None, optional
        Compute device.

    Returns
    -------
    int
        Estimated number of training epochs.
    """

    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
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

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )

    accur = []
    converged = False

    epoch = min(200, max_training_epochs)
    for _ in range(epoch):
        epoch_loss = train_one_epoch(model, loader, criterion, optimizer)
        accur.append(epoch_loss)
    maxep = epoch

    while not converged and epoch < max_training_epochs:
        for _ in range(10):
            epoch_loss = train_one_epoch(model, loader, criterion, optimizer)
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
            warnings.warn(
                f"Reached the maximum of {max_training_epochs} training epochs",
                ConvergenceWarning,
                stacklevel=2,
            )
            maxep = max_training_epochs

    return maxep


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
    verbose=True,
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
    verbose : bool, optional
        Print progress and the selected values.

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

    # Seed dropout during the search, and the bias draw below, so the chosen
    # hyperparameters are reproducible for a given seed.
    if seed is not None:
        torch.manual_seed(seed)

    if verbose:
        print("Determining learning rate...")
    key = "tree_0"
    weights, biases = build_tree_weights_and_biases(ttn, key, seed=seed)
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

    if verbose:
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

    if verbose:
        print("Optimal learning rate: ", lr)
        print("Optimal # epochs: ", opt_epochs)
        print("Optimal batch size: ", batch_size)

    return (batch_size, lr, opt_epochs)


def get_min_max(x, y, n_classes):
    """Compute per-dimension minimum and maximum values for inputs and outputs.

    Parameters
    ----------
    x : ndarray
        Input feature matrix.
    y : ndarray
        Output targets.
    n_classes : int
        Number of output targets/classes.

    Returns
    -------
    tuple[list[numpy.ndarray], list[numpy.ndarray]]
        Tuple ``(input_minmax, output_minmax)`` where each element is
        ``[min_values, max_values]``.
    """
    input_min = np.min(x, axis=0)
    input_max = np.max(x, axis=0)
    if n_classes == 1:
        y = y.reshape(-1, 1)
    output_min = np.min(y, axis=0)
    output_max = np.max(y, axis=0)

    return [input_min, input_max], [output_min, output_max]


def get_weights_and_biases(model):
    """Extract dense layer weights and biases from a trained model.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters are extracted.

    Returns
    -------
    tuple[list[numpy.ndarray], list[numpy.ndarray]]
        Tuple ``(dense_weights, dense_biases)`` in model parameter order.
    """
    dense_weights = []
    dense_biases = []

    for name, param in model.named_parameters():
        if "weight" in name:
            dense_weights.append(param.data)
        elif "bias" in name:
            dense_biases.append(param.data)

    # Copy, or on CPU the arrays keep changing with the model during training
    dense_weights = [w.detach().cpu().numpy().copy() for w in dense_weights]
    dense_biases = [b.detach().cpu().numpy().copy() for b in dense_biases]

    return dense_weights, dense_biases


def train_single_tree(
    model, loader, criterion, optimizer, xtest, ytest, epochs, eval_every=1
):
    """Train a single tree-mapped network and track train/validation losses.

    Parameters
    ----------
    model : nn.Module
        Network initialized from one mapped tree.
    loader : BatchLoader
        Training dataloader.
    criterion : callable
        Loss function used for training and validation.
    optimizer : torch.optim.Optimizer
        Optimizer used during training.
    xtest : torch.Tensor
        Validation input features.
    ytest : torch.Tensor
        Validation targets.
    epochs : int
        Number of training epochs.
    eval_every : int, optional
        Evaluate the full validation set every this many epochs (default 1).
        Values > 1 skip intermediate evaluations, reusing the last computed
        val loss, which substantially reduces overhead for large epoch counts.

    Returns
    -------
    tuple[nn.Module, list[float], list[float]]
        Tuple ``(model, train_history, valid_history)`` where histories hold
        per-epoch loss values.
    """

    train_history = []
    valid_history = []
    val_loss = float("nan")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, loader, criterion, optimizer)
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            val_loss = evaluate(model, xtest, ytest, criterion)

        train_history.append(train_loss)
        valid_history.append(val_loss)

    return model, train_history, valid_history


def save_tree_model(model, tree_idx, ttn, model_dir):
    """Persist one trained tree-model checkpoint to disk.

    Parameters
    ----------
    model : nn.Module
        Trained model to serialize.
    tree_idx : int
        Zero-based index of the tree in the ensemble.
    ttn : dict
        Tree-to-network mapping dictionary containing ``network_shape``.
    model_dir : pathlib.Path
        Directory where the checkpoint is written.
    """
    save_path = model_dir / f"tree_{tree_idx}.pt"

    network_shape_by_tree = ttn["network_shape"]
    tree_key = f"tree_{tree_idx}"

    if isinstance(network_shape_by_tree, dict):
        if tree_key in network_shape_by_tree:
            tree_network_shape = network_shape_by_tree[tree_key]
        else:
            # Backward-compatible fallback for legacy int-keyed dictionaries.
            tree_network_shape = network_shape_by_tree[tree_idx]
    else:
        # Backward-compatible fallback for list/array-like storage.
        tree_network_shape = network_shape_by_tree[tree_idx]

    torch.save(
        {
            "state_dict": model.state_dict(),
            "network_shape": tree_network_shape,
        },
        save_path,
    )


def save_experiment_metadata(model_dir, config, history):
    """Save run configuration and loss history metadata.

    Parameters
    ----------
    model_dir : pathlib.Path
        Directory where metadata files are written.
    config : dict
        Experiment settings used for training.
    history : dict
        Training/validation history arrays.
    """
    torch.save(config, model_dir / "config.pt")
    torch.save(history, model_dir / "history.pt")


def get_unique_model_name(model_path):
    """Create a unique model directory path by appending a numeric suffix.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Desired base path for model output directory.

    Returns
    -------
    pathlib.Path
        Newly created unique directory path.
    """

    model_path = Path(model_path)
    unique_path = model_path

    counter = 0
    while unique_path.exists():
        counter += 1
        fcounter = str(counter).zfill(2)
        unique_path = model_path.with_name(f"{model_path.name}_{fcounter}")

    # Create the unique directory
    unique_path.mkdir(parents=True, exist_ok=False)
    return unique_path


def torch_dropout_regression(
    regression,
    ttn,
    xscale,
    yscale,
    x,
    y,
    ntrees,
    lr,
    n_epochs,
    batch_size,
    dropout_keep_prob,
    weight_decay,
    **kwargs,
):
    """Train DJINN tree-initialized neural networks with PyTorch dropout.

    Parameters
    ----------
    regression : bool
        Whether the task is regression or classification.
    ttn : dict
        Dictionary returned from tree-to-network mapping.
    xscale : object
        Fitted scaler used to transform input features.
    yscale : object
        Fitted scaler used to transform targets for regression.
    x : ndarray
        Input features.
    y : ndarray
        Output targets.
    ntrees : int
        Number of tree-mapped networks to train.
    lr : float
        Adam learning rate.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    dropout_keep_prob : float
        Probability of keeping hidden units active during dropout.
    weight_decay : float
        L2 regularization strength for optimizer updates.
    **kwargs : dict
        Optional keyword arguments including ``save_model``, ``save_files``,
        ``model_path``, ``device``, and ``seed``.

    Returns
    -------
    dict
        Trained network information including initial/final weights and biases,
        input/output min-max values, and train/validation loss history.
    """
    # Kwargs
    save_model = kwargs.get("save_model", True)
    save_files = kwargs.get("save_files", True)
    model_path = kwargs.get("model_path", "djinn_model")
    device = kwargs.get("device", "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    # Set Seed
    seed = kwargs.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_classes = ttn["n_out"]
    if n_classes == 1:
        y = y.reshape(-1, 1)

    # save min/max values for python-only djinn eval
    input_minmax, output_minmax = get_min_max(x, y, n_classes)

    # create dict/arrays to save network info
    nninfo = {
        "input_minmax": input_minmax,
        "output_minmax": output_minmax,
        "initial_weights": {},
        "initial_biases": {},
        "final_weights": {},
        "final_biases": {},
        "models": {},  # live nn.Module objects, keyed by tree index
    }

    # Scale data and split into train/test sets
    xtrain, xtest, ytrain, ytest = scale_data(
        x, y, xscale, yscale, regression, seed, n_classes, test=True
    )

    if regression and n_classes == 1:
        if ytrain.ndim == 1:
            ytrain = ytrain.unsqueeze(1)
        if ytest.ndim == 1:
            ytest = ytest.unsqueeze(1)

    model_dir = None
    if save_model or save_files:
        model_dir = get_unique_model_name(model_path)
        nninfo["model_dir"] = str(model_dir)

    if not regression:
        ytrain, ytest = ytrain.argmax(dim=1), ytest.argmax(dim=1)

    keys = list(ttn["weights"])
    # Pass seed and idx so each tree's bias draw is isolated from
    # any numpy state consumed by earlier operations.
    tree_params = [
        build_tree_weights_and_biases(ttn, key, seed=seed, tree_idx=idx)
        for idx, key in enumerate(keys)
    ]
    for (weights, biases), key in zip(tree_params, keys):
        model = MultiLayerPerceptron(weights, biases, dropout_keep_prob)
        dense_weights, dense_biases = get_weights_and_biases(model)
        nninfo["initial_weights"][key] = dense_weights
        nninfo["initial_biases"][key] = dense_biases

    # Trees with the same number of layers are trained together
    groups = {}
    for idx, (weights, _) in enumerate(tree_params):
        groups.setdefault(len(weights), []).append(idx)

    all_train_history = [None] * len(keys)
    all_valid_history = [None] * len(keys)
    trained = {}
    for idxs in groups.values():
        layer_keys, stacked_w, stacked_b = stack_networks(
            [tree_params[i][0] for i in idxs], [tree_params[i][1] for i in idxs], device
        )
        generators = []
        for idx in idxs:
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed * 1000 + idx)
            generators.append(generator)
        optimizer = StackedAdam(
            stacked_w + stacked_b, lr, weight_decay, eps=1e-7  # TF default
        )
        train_hist, valid_hist = train_stacked(
            stacked_w,
            stacked_b,
            xtrain,
            ytrain,
            regression,
            generators,
            batch_size,
            n_epochs,
            optimizer,
            dropout_p=1.0 - dropout_keep_prob,
            xval=xtest,
            yval=ytest,
            eval_every=kwargs.get("eval_every", 1),
        )
        for j, idx in enumerate(idxs):
            shapes = {k: tree_params[idx][0][k].shape for k in layer_keys}
            trained[idx] = unstack_network(layer_keys, stacked_w, stacked_b, j, shapes)
            all_train_history[idx] = train_hist[j].tolist()
            all_valid_history[idx] = valid_hist[j].tolist()

    for idx, key in enumerate(keys):
        weights, biases = trained[idx]
        model = MultiLayerPerceptron(weights, biases, dropout_keep_prob).to(device)
        dense_weights, dense_biases = get_weights_and_biases(model)
        nninfo["final_weights"][key] = dense_weights
        nninfo["final_biases"][key] = dense_biases

        # Always keep the live model in memory regardless of save_model
        nninfo["models"][idx] = model
        if save_model:
            save_tree_model(model, idx, ttn, model_dir)

    # Save experiment metadata
    if len(all_train_history) == 1:
        nninfo["train_cost"] = np.array(all_train_history[0])
        nninfo["valid_cost"] = np.array(all_valid_history[0])
    else:
        nninfo["train_cost"] = np.array(all_train_history)
        nninfo["valid_cost"] = np.array(all_valid_history)

    if save_files:
        config = {
            "regression": regression,
            "n_classes": n_classes,
            "ntrees": ntrees,
            "lr": lr,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "dropout_keep_prob": dropout_keep_prob,
            "weight_decay": weight_decay,
            "seed": seed,
        }

        history = {
            "train_loss": np.array(all_train_history),
            "valid_loss": np.array(all_valid_history),
        }
        save_experiment_metadata(model_dir, config, history)

    return nninfo


def load_tree_data(xscale, yscale, x, y, regression, batch_size, device):
    """Scale training data and build a retraining dataloader.

    Parameters
    ----------
    xscale : object
        Fitted scaler used to transform input features.
    yscale : object
        Fitted scaler used to transform regression targets.
    x : ndarray
        Input feature matrix.
    y : ndarray
        Training targets.
    regression : bool
        Whether the task is regression or classification.
    batch_size : int
        Mini-batch size.
    device : torch.device
        Device where tensors are materialized.

    Returns
    -------
    BatchLoader
        Shuffled dataloader for retraining.
    """
    # Scale training data
    y = np.asarray(y)
    if regression:
        n_classes = y.shape[1] if (y.ndim > 1 and y.shape[1] > 1) else 1
    else:
        n_classes = np.unique(y).size

    xtrain = xscale.transform(x)
    if regression and n_classes == 1:
        ytrain = yscale.transform(y.reshape(-1, 1))
    elif regression:
        ytrain = yscale.transform(y)
    else:
        ytrain = F.one_hot(
            torch.as_tensor(y.flatten(), dtype=torch.long), num_classes=n_classes
        ).to(dtype=torch.float32)

    loader = prepare_dataloader(
        xtrain,
        ytrain,
        regression,
        batch_size,
        device,
    )

    return loader


def load_tree_model(checkpoint_path, device, dropout_keep_prob, tree_idx, verbose=True):
    """Restore a saved tree checkpoint as a PyTorch model.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Path to the ``tree_*.pt`` checkpoint.
    device : torch.device
        Device used to load tensors and construct the model.
    dropout_keep_prob : float
        Keep probability used when rebuilding dropout layers.
    tree_idx : int
        Zero-based tree index for logging.
    verbose : bool, optional
        Print a message once the tree is restored.

    Returns
    -------
    tuple[nn.Module, dict]
        Tuple of ``(model, network_shape)`` restored from checkpoint.
    """
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

    # Rebuild model from saved structure. Older checkpoints store
    # a dict with explicit weights/biases under "network_shape" while
    # newer checkpoints may store only an architecture array.
    network_shape = checkpoint["network_shape"]

    if isinstance(network_shape, dict) and "weights" in network_shape:
        weights = network_shape["weights"]
        biases = network_shape["biases"]
    else:
        state_dict = checkpoint["state_dict"]
        hidden_weight_keys = sorted(
            [
                k
                for k in state_dict.keys()
                if k.startswith("hidden_layers.") and k.endswith(".weight")
            ],
            key=lambda name: int(name.split(".")[1]),
        )

        weights = {}
        biases = {}

        for layer_num, weight_key in enumerate(hidden_weight_keys, start=1):
            bias_key = weight_key.replace(".weight", ".bias")
            w = state_dict[weight_key].detach().cpu().numpy()
            b = state_dict[bias_key].detach().cpu().numpy()
            weights[f"h{layer_num}"] = w.T
            biases[f"h{layer_num}"] = b

        out_w = state_dict["output_layer.weight"].detach().cpu().numpy()
        out_b = state_dict["output_layer.bias"].detach().cpu().numpy()
        weights["out"] = out_w.T
        biases["out"] = out_b

        network_shape = {
            "weights": weights,
            "biases": biases,
        }

    model = MultiLayerPerceptron(
        weights,
        biases,
        dropout_keep_prob,
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    if verbose:
        print(f"Tree {tree_idx} restored")
    return model, network_shape


def torch_continue_training(
    regression,
    xscale,
    yscale,
    x,
    y,
    ntrees,
    lr,
    n_epochs,
    batch_size,
    dropout_keep_prob,
    model_dir,
    model_name=None,
    weight_decay=0.0,
    seed=None,
    device=None,
    verbose=True,
):
    """Continue training previously saved DJINN PyTorch models.

    Parameters
    ----------
    regression : bool
        Regression or classification task.
    xscale : fitted scaler
        Input scaler.
    yscale : fitted scaler
        Output scaler (regression only).
    x : ndarray
        Training inputs.
    y : ndarray
        Training targets.
    ntrees : int
        Number of tree models in ensemble.
    lr : float
        Learning rate.
    n_epochs : int
        Additional epochs to train.
    batch_size : int
        Mini-batch size.
    dropout_keep_prob : float
        Keep probability for dropout.
    model_dir : str or Path
        Directory containing saved tree_*.pt models.
    model_name : str or None
        Optional model name used when writing retraining metadata file.
    weight_decay : float
        L2 regularization.
    seed : int or None
        Random seed.
    device : torch.device, str, or None
        Device to train on.
    verbose : bool, optional
        Print progress for each tree.

    Returns
    -------
    None
        Re-saves tree checkpoints and writes retraining metadata.
    """

    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(model_dir)

    if seed is not None:
        torch.manual_seed(seed)

    loader = load_tree_data(xscale, yscale, x, y, regression, batch_size, device)

    nninfo = {
        "weights": {},
        "biases": {},
        "initial_weights": {},
        "initial_biases": {},
    }

    # Continue training each tree
    for tree_idx in range(ntrees):

        # Load checkpoint for this tree
        checkpoint_path = model_dir / f"tree_{tree_idx}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load the tree model
        model, network_shape = load_tree_model(
            checkpoint_path, device, dropout_keep_prob, tree_idx, verbose
        )
        dense_weights, dense_biases = get_weights_and_biases(model)
        nninfo["initial_weights"][f"tree{tree_idx}"] = dense_weights
        # nninfo["initial_biases"][f"tree{tree_idx}"] = dense_biases

        # Train model for additional epochs
        _ = train_model(
            model,
            loader,
            regression,
            lr,
            weight_decay,
            epochs=n_epochs,
            early_stop_patience=None,
        )
        if verbose:
            print("Optimization Finished!")

        # Resave checkpoint
        torch.save(
            {
                "state_dict": model.state_dict(),
                "network_shape": network_shape,
            },
            checkpoint_path,
        )
        if verbose:
            print(f"Tree {tree_idx} resaved at {checkpoint_path}")

        # Save final weights/biases to metadata
        dense_weights, dense_biases = get_weights_and_biases(model)
        nninfo["weights"][f"tree{tree_idx}"] = dense_weights
        nninfo["biases"][f"tree{tree_idx}"] = dense_biases

    if model_name is None:
        model_name = model_dir.name

    with open(model_dir / f"retrained_nn_info_{model_name}.json", "w") as file:
        json.dump(nninfo, file, indent=4, cls=NumpyEncoder)
