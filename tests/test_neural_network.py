"""Unit tests for DJINN neural-network training and model lifecycle helpers."""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

import djinn.neural_network as nnf


def make_regression_data(n=24):
    x = np.linspace(0.0, 1.0, n * 2).reshape(n, 2).astype(np.float32)
    y = (2.0 * x[:, 0] - 0.5 * x[:, 1] + 0.1).astype(np.float32).reshape(-1, 1)
    return x, y


def make_classification_data(n=24):
    x = np.linspace(-1.0, 1.0, n * 2).reshape(n, 2).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
    return x, y


def make_ttn_regression():
    return {
        "n_in": 2,
        "n_out": 1,
        "network_shape": {"tree_0": [2, 3, 1]},
        "weights": {
            "tree_0": [
                np.array([[0.2, -0.1], [0.1, 0.3], [0.5, -0.4]], dtype=np.float32),
                np.array([[0.7, -0.2, 0.1]], dtype=np.float32),
            ]
        },
        "biases": {"tree_0": []},
    }


def make_ttn_two_widths(n_out=1):
    """Two trees with different hidden widths."""
    return {
        "n_in": 2,
        "n_out": n_out,
        "network_shape": {"tree_0": [2, 3, n_out], "tree_1": [2, 5, n_out]},
        "weights": {
            "tree_0": [
                np.array([[0.2, -0.1], [0.1, 0.3], [0.5, -0.4]], dtype=np.float32),
                np.full((n_out, 3), 0.3, dtype=np.float32),
            ],
            "tree_1": [
                np.array(
                    [[0.1, 0.2], [0.0, -0.3], [0.2, 0.4], [0.3, 0.1], [-0.2, 0.5]],
                    dtype=np.float32,
                ),
                np.full((n_out, 5), -0.2, dtype=np.float32),
            ],
        },
        "biases": {"tree_0": [], "tree_1": []},
    }


def make_ttn_classification():
    return {
        "n_in": 2,
        "n_out": 2,
        "network_shape": {"tree_0": [2, 3, 2]},
        "weights": {
            "tree_0": [
                np.array([[0.1, 0.2], [0.0, -0.3], [0.2, 0.4]], dtype=np.float32),
                np.array([[0.3, -0.1, 0.2], [-0.2, 0.4, 0.1]], dtype=np.float32),
            ]
        },
        "biases": {"tree_0": []},
    }


def test_scale_data_regression_shapes():
    x, y = make_regression_data()
    xscale = MinMaxScaler().fit(x)
    yscale = MinMaxScaler().fit(y)

    xtrain, xtest, ytrain, ytest = nnf.scale_data(
        x, y, xscale, yscale, True, seed=0, n_classes=1, test=True
    )

    assert isinstance(xtrain, torch.Tensor)
    assert isinstance(ytrain, torch.Tensor)
    assert xtrain.shape[1] == 2
    assert ytrain.ndim == 1
    assert xtrain.shape[0] + xtest.shape[0] == x.shape[0]
    assert ytrain.shape[0] + ytest.shape[0] == y.shape[0]


def test_scale_data_classification_one_hot_shapes():
    x, y = make_classification_data()
    xscale = MinMaxScaler().fit(x)

    xtrain, xtest, ytrain, ytest = nnf.scale_data(
        x, y, xscale, None, False, seed=0, n_classes=2, test=True
    )

    assert isinstance(ytrain, torch.Tensor)
    assert isinstance(ytest, torch.Tensor)
    assert ytrain.ndim == 2
    assert ytest.ndim == 2
    assert ytrain.shape[1] == 2
    assert ytest.shape[1] == 2


def test_mlp_forward_output_shape():
    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    model = nnf.MultiLayerPerceptron(weights, biases, dropout_keep_prob=1.0)
    out = model(torch.randn(5, 2))
    assert out.shape == (5, 1)


def test_build_tree_weights_and_biases_structure():
    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    assert set(weights.keys()) == {"h1", "out"}
    assert set(biases.keys()) == {"h1", "out"}
    assert weights["h1"].shape == (2, 3)
    assert weights["out"].shape == (3, 1)


def test_prepare_dataloader_regression_and_classification():
    device = torch.device("cpu")
    xr, yr = make_regression_data()
    xc, yc = make_classification_data()

    reg_loader = nnf.prepare_dataloader(xr, yr, True, batch_size=8, device=device)
    xb, yb = next(iter(reg_loader))
    assert xb.dtype == torch.float32
    assert yb.dtype == torch.float32

    y_one_hot = np.eye(2, dtype=np.float32)[yc]
    cls_loader = nnf.prepare_dataloader(
        xc, y_one_hot, False, batch_size=8, device=device
    )
    xb2, yb2 = next(iter(cls_loader))
    assert xb2.dtype == torch.float32
    assert yb2.dtype == torch.int64


def test_prepare_dataloader_samples_full_batches_like_tf():
    device = torch.device("cpu")
    x = np.arange(21, dtype=np.float32).reshape(-1, 1)
    y = np.arange(21, dtype=np.float32)

    loader = nnf.prepare_dataloader(x, y, True, batch_size=4, device=device, seed=0)
    batches = [xb for xb, _ in loader]
    # TF DJINN draws len // batch_size full batches with replacement each epoch
    assert len(batches) == 21 // 4
    assert all(len(xb) == 4 for xb in batches)

    again = nnf.prepare_dataloader(x, y, True, batch_size=4, device=device, seed=0)
    assert all(torch.equal(a, b) for (a, _), b in zip(again, batches))


def test_stack_networks_pads_with_zeros():
    ttn = make_ttn_two_widths()
    params = [
        nnf.build_tree_weights_and_biases(ttn, key, seed=0, tree_idx=i)
        for i, key in enumerate(ttn["weights"])
    ]
    keys, stacked_w, stacked_b = nnf.stack_networks(
        [w for w, _ in params], [b for _, b in params], torch.device("cpu")
    )
    assert keys == ["h1", "out"]
    assert stacked_w[0].shape == (2, 2, 5)
    assert torch.all(stacked_w[0][0, :, 3:] == 0)
    assert torch.all(stacked_w[1][0, 3:] == 0)

    shapes = {k: params[0][0][k].shape for k in keys}
    weights, biases = nnf.unstack_network(keys, stacked_w, stacked_b, 0, shapes)
    for key in keys:
        assert np.array_equal(weights[key], params[0][0][key])
        assert np.allclose(biases[key], params[0][1][key])


@pytest.mark.parametrize("regression", [True, False])
def test_stacked_training_matches_separate_networks(regression):
    device = torch.device("cpu")
    n_out = 1 if regression else 2
    ttn = make_ttn_two_widths(n_out)
    if regression:
        x, y = make_regression_data(40)
        ytrain = torch.as_tensor(y)
    else:
        x, y = make_classification_data(40)
        ytrain = torch.as_tensor(y)
    keys = list(ttn["weights"])
    params = [
        nnf.build_tree_weights_and_biases(ttn, key, seed=0, tree_idx=i)
        for i, key in enumerate(keys)
    ]
    epochs, lr, weight_decay = 2, 1e-2, 1e-3

    separate = []
    for i, (weights, biases) in enumerate(params):
        loader = nnf.prepare_dataloader(x, y, regression, 8, device, seed=i)
        model = nnf.MultiLayerPerceptron(weights, biases, dropout_keep_prob=1.0)
        criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-7
        )
        for _ in range(epochs):
            nnf.train_one_epoch(model, loader, criterion, optimizer)
        separate.append(model)

    layer_keys, stacked_w, stacked_b = nnf.stack_networks(
        [w for w, _ in params], [b for _, b in params], device
    )
    optimizer = nnf.StackedAdam(stacked_w + stacked_b, lr, weight_decay, eps=1e-7)
    generators = [torch.Generator().manual_seed(i) for i in range(len(params))]
    history, _ = nnf.train_stacked(
        stacked_w,
        stacked_b,
        torch.as_tensor(x),
        ytrain,
        regression,
        generators,
        8,
        epochs,
        optimizer,
    )
    assert history.shape == (2, epochs)

    for i, model in enumerate(separate):
        shapes = {k: params[i][0][k].shape for k in layer_keys}
        weights, biases = nnf.unstack_network(
            layer_keys, stacked_w, stacked_b, i, shapes
        )
        layers = [*model.hidden_layers, model.output_layer]
        for key, layer in zip(layer_keys, layers):
            expected = layer.weight.detach().numpy().T
            assert np.allclose(weights[key], expected, atol=1e-6)
            assert np.allclose(biases[key], layer.bias.detach().numpy(), atol=1e-6)


def test_stacked_adam_matches_torch_adam_per_network():
    torch.manual_seed(0)
    lrs = [1e-3, 1e-2]
    start = torch.randn(2, 3, 4)
    stacked = start.clone().requires_grad_()
    optimizer = nnf.StackedAdam(
        [stacked], torch.tensor(lrs).view(-1, 1, 1), weight_decay=0.1
    )
    singles = [start[i].clone().requires_grad_() for i in range(2)]
    torch_opts = [
        torch.optim.Adam([p], lr=lr, weight_decay=0.1, eps=1e-8)
        for p, lr in zip(singles, lrs)
    ]
    for _ in range(5):
        grad = torch.randn(2, 3, 4)
        stacked.grad = grad.clone()
        optimizer.step()
        for i, (p, opt) in enumerate(zip(singles, torch_opts)):
            p.grad = grad[i].clone()
            opt.step()
    for i, p in enumerate(singles):
        assert torch.allclose(stacked[i], p, atol=1e-6)


def test_resolve_device(monkeypatch):
    assert nnf.resolve_device("cpu") == torch.device("cpu")
    assert nnf.resolve_device(torch.device("cpu")) == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert nnf.resolve_device("auto") == torch.device("cpu")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert nnf.resolve_device("auto") == torch.device("mps")


def test_train_one_epoch_returns_float_loss():
    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    model = nnf.MultiLayerPerceptron(weights, biases, dropout_keep_prob=1.0)

    x, y = make_regression_data()
    loader = nnf.prepare_dataloader(
        x, y, True, batch_size=8, device=torch.device("cpu")
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = nnf.train_one_epoch(model, loader, criterion, optimizer)
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_evaluate_returns_float_loss():
    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    model = nnf.MultiLayerPerceptron(weights, biases, dropout_keep_prob=1.0)

    x, y = make_regression_data()
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    loss = nnf.evaluate(model, xt, yt, nn.MSELoss())
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_train_model_returns_epoch_losses():
    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    model = nnf.MultiLayerPerceptron(weights, biases, dropout_keep_prob=1.0)
    x, y = make_regression_data()
    loader = nnf.prepare_dataloader(
        x, y, True, batch_size=8, device=torch.device("cpu")
    )

    losses = nnf.train_model(
        model,
        loader,
        regression=True,
        lr=1e-3,
        weight_decay=0.0,
        epochs=3,
        early_stop_patience=None,
    )
    assert isinstance(losses, list)
    assert len(losses) == 3
    assert all(np.isfinite(v) for v in losses)


def test_get_learning_rate_runs():
    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    x, y = make_regression_data()

    lr = nnf.get_learning_rate(
        True,
        weights,
        biases,
        x,
        y,
        dropout_keep_prob=1.0,
        batch_size=8,
        weight_decay=0.0,
        device=torch.device("cpu"),
    )
    assert isinstance(lr, float)
    assert 1e-4 <= lr <= 1e-2


def test_find_optimal_epochs_runs(monkeypatch):
    monkeypatch.setattr(nnf, "train_one_epoch", lambda *a, **k: 1.0)

    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    x, y = make_regression_data()

    epochs = nnf.find_optimal_epochs(
        True,
        weights,
        biases,
        x,
        y,
        dropout_keep_prob=1.0,
        lr=1e-3,
        batch_size=8,
        max_training_epochs=260,
        device=torch.device("cpu"),
    )
    assert isinstance(epochs, int)
    assert epochs >= 200


def test_get_hyperparams_with_patched_search(monkeypatch):
    monkeypatch.setattr(nnf, "get_learning_rate", lambda *a, **k: 0.005)
    monkeypatch.setattr(nnf, "find_optimal_epochs", lambda *a, **k: 42)

    x, y = make_regression_data()
    xscale = MinMaxScaler().fit(x)
    yscale = MinMaxScaler().fit(y)
    ttn = make_ttn_regression()

    batch_size, lr, epochs = nnf.get_hyperparams(
        True,
        ttn,
        xscale,
        yscale,
        x,
        y,
        dropout_keep_prob=1.0,
        weight_decay=0.0,
        seed=0,
        device=torch.device("cpu"),
    )
    assert isinstance(batch_size, int)
    assert lr == 0.005
    assert epochs == 42


def test_get_min_max_values():
    x, y = make_regression_data()
    input_minmax, output_minmax = nnf.get_min_max(x, y, n_classes=1)
    assert np.allclose(input_minmax[0], np.min(x, axis=0))
    assert np.allclose(input_minmax[1], np.max(x, axis=0))
    assert np.allclose(output_minmax[0], np.min(y, axis=0))
    assert np.allclose(output_minmax[1], np.max(y, axis=0))


def test_get_weights_and_biases_numpy_arrays():
    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    model = nnf.MultiLayerPerceptron(weights, biases, dropout_keep_prob=1.0)
    dense_w, dense_b = nnf.get_weights_and_biases(model)

    assert all(isinstance(w, np.ndarray) for w in dense_w)
    assert all(isinstance(b, np.ndarray) for b in dense_b)
    assert len(dense_w) == 2
    assert len(dense_b) == 2


def test_get_weights_and_biases_returns_copies():
    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    model = nnf.MultiLayerPerceptron(weights, biases, dropout_keep_prob=1.0)
    dense_w, dense_b = nnf.get_weights_and_biases(model)
    before_w = [w.copy() for w in dense_w]
    before_b = [b.copy() for b in dense_b]

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    assert all(np.array_equal(w, b) for w, b in zip(dense_w, before_w))
    assert all(np.array_equal(w, b) for w, b in zip(dense_b, before_b))


def test_train_single_tree_histories(monkeypatch):
    monkeypatch.setattr(nnf, "train_one_epoch", lambda *a, **k: 0.75)
    monkeypatch.setattr(nnf, "evaluate", lambda *a, **k: 0.5)

    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    model = nnf.MultiLayerPerceptron(weights, biases, dropout_keep_prob=1.0)

    x, y = make_regression_data()
    loader = nnf.prepare_dataloader(
        x, y, True, batch_size=8, device=torch.device("cpu")
    )
    xt = torch.as_tensor(x[:6], dtype=torch.float32)
    yt = torch.as_tensor(y[:6], dtype=torch.float32)

    out_model, train_hist, valid_hist = nnf.train_single_tree(
        model,
        loader,
        nn.MSELoss(),
        torch.optim.Adam(model.parameters(), lr=1e-3),
        xt,
        yt,
        epochs=4,
    )
    assert out_model is model
    assert train_hist == [0.75] * 4
    assert valid_hist == [0.5] * 4


def test_save_tree_model_writes_checkpoint(tmp_path):
    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    model = nnf.MultiLayerPerceptron(weights, biases, dropout_keep_prob=1.0)

    compat_ttn = {"network_shape": {0: {"weights": weights, "biases": biases}}}
    nnf.save_tree_model(model, tree_idx=0, ttn=compat_ttn, model_dir=tmp_path)
    assert (tmp_path / "tree_0.pt").exists()


def test_save_experiment_metadata_writes_files(tmp_path):
    nnf.save_experiment_metadata(
        tmp_path,
        config={"a": 1},
        history={"train_loss": np.array([1.0]), "valid_loss": np.array([2.0])},
    )
    assert (tmp_path / "config.pt").exists()
    assert (tmp_path / "history.pt").exists()


def test_get_unique_model_name_creates_suffix(tmp_path):
    base = tmp_path / "djinn_model"
    base.mkdir()
    unique = nnf.get_unique_model_name(base)
    assert unique.exists()
    assert unique.name.startswith("djinn_model_")


def test_torch_dropout_regression_returns_nninfo(tmp_path):
    x, y = make_regression_data()
    xscale = MinMaxScaler().fit(x)
    yscale = MinMaxScaler().fit(y)
    ttn = make_ttn_regression()

    out = nnf.torch_dropout_regression(
        True,
        ttn,
        xscale,
        yscale,
        x,
        y,
        ntrees=1,
        lr=1e-3,
        n_epochs=2,
        batch_size=8,
        dropout_keep_prob=1.0,
        weight_decay=0.0,
        save_model=False,
        save_files=False,
        model_path=str(tmp_path / "model"),
        device="cpu",
        seed=0,
    )

    assert "initial_weights" in out
    assert "final_weights" in out
    assert "train_cost" in out
    assert "valid_cost" in out
    assert "tree_0" in out["initial_weights"]


def test_load_tree_data_regression_and_classification():
    x, y = make_regression_data()
    xscale = MinMaxScaler().fit(x)
    yscale = MinMaxScaler().fit(y)
    loader = nnf.load_tree_data(
        xscale, yscale, x, y, True, batch_size=8, device=torch.device("cpu")
    )
    xb, yb = next(iter(loader))
    assert xb.shape[1] == x.shape[1]
    assert yb.ndim == 2

    xc, yc = make_classification_data()
    xscale_c = MinMaxScaler().fit(xc)
    loader_c = nnf.load_tree_data(
        xscale_c, yscale, xc, yc, False, batch_size=8, device=torch.device("cpu")
    )
    xb2, yb2 = next(iter(loader_c))
    assert xb2.shape[1] == xc.shape[1]
    assert yb2.dtype == torch.int64


def test_load_tree_model_restores_checkpoint(tmp_path):
    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    model = nnf.MultiLayerPerceptron(weights, biases, dropout_keep_prob=1.0)

    ckpt = tmp_path / "tree_0.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "network_shape": {"weights": weights, "biases": biases},
        },
        ckpt,
    )

    out_model, shape = nnf.load_tree_model(
        ckpt, torch.device("cpu"), dropout_keep_prob=1.0, tree_idx=0
    )
    assert isinstance(out_model, nn.Module)
    assert "weights" in shape and "biases" in shape


def test_torch_continue_training_runs_and_writes_metadata(tmp_path, monkeypatch):
    x, y = make_regression_data()
    xscale = MinMaxScaler().fit(x)
    yscale = MinMaxScaler().fit(y)
    ttn = make_ttn_regression()
    weights, biases = nnf.build_tree_weights_and_biases(ttn, "tree_0")
    model = nnf.MultiLayerPerceptron(weights, biases, dropout_keep_prob=1.0)

    checkpoint_path = tmp_path / "tree_0.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "network_shape": {"weights": weights, "biases": biases},
        },
        checkpoint_path,
    )

    def safe_json_dump(obj, fp, indent=4, **kwargs):
        serializable = json.loads(
            json.dumps(obj, default=lambda o: np.asarray(o).tolist())
        )
        fp.write(json.dumps(serializable, indent=indent))

    monkeypatch.setattr(nnf.json, "dump", safe_json_dump)

    nnf.torch_continue_training(
        regression=True,
        xscale=xscale,
        yscale=yscale,
        x=x,
        y=y,
        ntrees=1,
        lr=1e-3,
        n_epochs=1,
        batch_size=8,
        dropout_keep_prob=1.0,
        model_dir=tmp_path,
        model_name="unit",
        weight_decay=0.0,
        seed=0,
        device=torch.device("cpu"),
    )

    assert checkpoint_path.exists()
    assert (tmp_path / "retrained_nn_info_unit.json").exists()
