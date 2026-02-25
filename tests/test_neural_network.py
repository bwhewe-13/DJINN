import json

import numpy as np
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


def test_get_learning_rate_runs(monkeypatch):
    def fake_train_model(*args, **kwargs):
        lr = args[3]
        return [1.0 + float(lr)] * 100

    monkeypatch.setattr(nnf, "train_model", fake_train_model)

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

    def safe_json_dump(obj, fp, indent=4):
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
