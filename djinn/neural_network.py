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

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


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
        ytrain = F.one_hot(ytrain.flatten(), num_classes=torch.unique(ytrain).numel())
        ytest = F.one_hot(ytest.flatten(), num_classes=torch.unique(ytest).numel())

    xtrain_tensor = torch.tensor(xtrain, dtype=torch.float32)
    xtest_tensor = torch.tensor(xtest, dtype=torch.float32)
    ytrain_tensor = torch.tensor(ytrain, dtype=torch.float32)
    ytest_tensor = torch.tensor(ytest, dtype=torch.float32)

    if test:
        return xtrain_tensor, xtest_tensor, ytrain_tensor, ytest_tensor
    return xtrain_tensor, ytrain_tensor


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

    dataset = TensorDataset(xtrain, ytrain)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_one_epoch(model, loader, criterion, optimizer):
    """Run one optimization epoch over the training loader.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    loader : torch.utils.data.DataLoader
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
        total_loss += loss.item()

    return total_loss / len(loader)


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

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )

    accur = []
    epoch = 0
    converged = False

    epoch = 200
    for _ in range(epoch):
        epoch_loss = train_one_epoch(model, loader, criterion, optimizer)
        accur.append(epoch_loss)

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
            print("Warning: Reached max # training epochs:", max_training_epochs)
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

    dense_weights = [w.detach().cpu().numpy() for w in dense_weights]
    dense_biases = [b.detach().cpu().numpy() for b in dense_biases]

    return dense_weights, dense_biases


def train_single_tree(model, loader, criterion, optimizer, xtest, ytest, epochs):
    """Train a single tree-mapped network and track train/validation losses.

    Parameters
    ----------
    model : nn.Module
        Network initialized from one mapped tree.
    loader : torch.utils.data.DataLoader
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

    Returns
    -------
    tuple[nn.Module, list[float], list[float]]
        Tuple ``(model, train_history, valid_history)`` where histories hold
        per-epoch loss values.
    """

    train_history = []
    valid_history = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, loader, criterion, optimizer)
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

    torch.save(
        {
            "state_dict": model.state_dict(),
            "network_shape": ttn["network_shape"][tree_idx],
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
    device = torch.device(kwargs.get("device", "cpu"))

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

    if save_model or save_files:
        model_dir = get_unique_model_name(model_path)

    all_train_history = []
    all_valid_history = []

    # loop through trees, training each network in ensemble
    for idx, keys in enumerate(ttn["weights"]):
        # Initialize model weights and biases from tree mapping
        weights, biases = build_tree_weights_and_biases(ttn, keys)
        loader = prepare_dataloader(xtrain, ytrain, regression, batch_size, device)

        # Create model and optimizer for this tree
        model = MultiLayerPerceptron(weights, biases, dropout_keep_prob).to(device)
        criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-7,  # TF default
        )

        # Save initial weights/biases
        dense_weights, dense_biases = get_weights_and_biases(model)
        nninfo["initial_weights"][keys] = dense_weights
        nninfo["initial_biases"][keys] = dense_biases

        # Train model and record history
        model, train_hist, valid_hist = train_single_tree(
            model, loader, criterion, optimizer, xtest, ytest, n_epochs
        )

        # Save final weights/biases
        dense_weights, dense_biases = get_weights_and_biases(model)
        nninfo["final_weights"][keys] = dense_weights
        nninfo["final_biases"][keys] = dense_biases

        if save_model:
            save_tree_model(model, idx, ttn, model_dir)

        all_train_history.append(train_hist)
        all_valid_history.append(valid_hist)

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
    torch.utils.data.DataLoader
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


def load_tree_model(checkpoint_path, device, dropout_keep_prob, tree_idx):
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

    # Rebuild model from saved structure
    network_shape = checkpoint["network_shape"]
    weights = network_shape["weights"]
    biases = network_shape["biases"]

    model = MultiLayerPerceptron(
        weights,
        biases,
        dropout_keep_prob,
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
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
    device : torch.device or None
        Device to train on.

    Returns
    -------
    None
        Re-saves tree checkpoints and writes retraining metadata.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            checkpoint_path, device, dropout_keep_prob, tree_idx
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
        print("Optimization Finished!")

        # Resave checkpoint
        torch.save(
            {
                "state_dict": model.state_dict(),
                "network_shape": network_shape,
            },
            checkpoint_path,
        )
        print(f"Tree {tree_idx} resaved at {checkpoint_path}")

        # Save final weights/biases to metadata
        dense_weights, dense_biases = get_weights_and_biases(model)
        nninfo["weights"][f"tree{tree_idx}"] = dense_weights
        nninfo["biases"][f"tree{tree_idx}"] = dense_biases

    if model_name is None:
        model_name = model_dir.name

    with open(model_dir / f"retrained_nn_info_{model_name}.json", "w") as file:
        json.dump(nninfo, file, indent=4)
