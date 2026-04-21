"""
run_and_collect.py — Run in EACH virtual environment to collect results.

This script trains DJINN across multiple seeds and datasets, then saves
a JSON file that the cross-environment comparison script will consume.

Usage:
    (tf-djinn) $ python run_and_collect.py --impl tf --out results_tf.json
    (pt-djinn) $ python run_and_collect.py --impl pt --out results_pt.json

Requirements (same in both envs):
    pip install scikit-learn numpy
"""

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Remove the repo root from sys.path before any other imports so that the
# djinn package installed in the active venv is found, rather than the
# local djinn/ source directory which would always shadow it.
_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _repo_root]

warnings.filterwarnings("ignore")

from djinn import djinn  # noqa: E402  (works in both envs)


def _make_model(impl, ntrees):
    """Construct a DJINN_Regressor for the selected implementation.

    TF:  DJINN_Regressor(ntrees, maxdepth, dropout_keep) - positional.
    PT:  DJINN_Regressor(n_trees=ntrees) - keyword.
    """
    if impl == "pt":
        return djinn.DJINN_Regressor(n_trees=ntrees)
    else:
        return djinn.DJINN_Regressor(ntrees)


def _fit(impl, model, X, y, epochs, seed):
    """Train the model using the correct API for each implementation.

    Both PT and TF use get_hyperparameters() to auto-tune learning rate and
    batch size, ensuring a fair comparison. PT suppresses file I/O during
    benchmarks via save_model=False / save_files=False.

    TF:  fit() takes no kwargs, so get_hyperparameters + train are called
         directly to control epochs and seed.
    """
    if impl == "pt":
        optimal = model.get_hyperparameters(X, y, seed=seed)
        model.train(
            X,
            y,
            epochs=epochs,
            learning_rate=optimal["learning_rate"],
            batch_size=optimal["batch_size"],
            save_model=False,
            save_files=False,
            seed=seed,
        )
    else:
        # TF: fit() takes no kwargs — it runs get_hyperparameters() internally.
        # To control epochs and seed, call get_hyperparameters + train directly.
        optimal = model.get_hyperparameters(X, y, random_state=seed)
        model.train(
            X,
            y,
            epochs=epochs,
            learn_rate=optimal["learn_rate"],
            batch_size=optimal["batch_size"],
            random_state=seed,
            display_step=101,  # > epochs to suppress per-epoch logging
        )


def _metrics(y_true, y_pred):
    """Compute regression metrics for true and predicted targets.

    Parameters
    ----------
    y_true : ndarray
        Ground-truth targets.
    y_pred : ndarray
        Predicted targets.

    Returns
    -------
    dict
        Dictionary containing ``mse``, ``mae``, and ``r2``.
    """
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def scale(X_train, X_test, y_train, y_test):
    """Min-max scale train/test features and targets.

    Parameters
    ----------
    X_train : ndarray
        Training feature matrix.
    X_test : ndarray
        Test feature matrix.
    y_train : ndarray
        Training targets.
    y_test : ndarray
        Test targets.

    Returns
    -------
    tuple
        Scaled ``X_train``, ``X_test``, ``y_train``, ``y_test``, and the
        fitted target scaler.
    """
    sx, sy = MinMaxScaler(), MinMaxScaler()
    return (
        sx.fit_transform(X_train),
        sx.transform(X_test),
        sy.fit_transform(y_train),
        sy.transform(y_test),
        sy,
    )


def _bma_samples(model, X_te, n_iters):
    """Run Bayesian prediction using whichever API is available.

    PT:  has bma_predict() which returns a dict with a 'predictions' array.
    TF:  only has bayesian_predict() which returns (lower, middle, upper, samples);
         extract and stack the per-tree predictions from the samples dict.
    """
    if hasattr(model, "bma_predict"):
        result = model.bma_predict(X_te, n_iters=n_iters)
        return np.array(result["predictions"])  # shape (n_iters*n_trees, n_test, n_out)
    else:
        # TF: bayesian_predict returns (lower, middle, upper, samples)
        _, _, _, samples = model.bayesian_predict(X_te, n_iters)
        # samples["predictions"] is a dict keyed by tree index or "treeN" string;
        # stack all draws into a single array
        tree_arrays = list(samples["predictions"].values())
        # each entry is a list of n_iters arrays of shape (n_test, n_out)
        stacked = np.concatenate(
            [np.array(draws) for draws in tree_arrays], axis=0
        )  # shape (n_iters * n_trees, n_test, n_out)
        return stacked


def run_regression(impl, seeds, ntrees=3, epochs=50):
    """Single-output regression on Diabetes dataset."""
    X, y = load_diabetes(return_X_y=True)
    y = y.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr, X_te, y_tr, y_te, sy = scale(X_tr, X_te, y_tr, y_te)

    records = []
    for seed in seeds:
        model = _make_model(impl, ntrees)
        t0 = time.perf_counter()
        _fit(impl, model, X_tr, y_tr, epochs=epochs, seed=seed)
        elapsed = time.perf_counter() - t0

        preds = sy.inverse_transform(model.predict(X_te))
        y_true = sy.inverse_transform(y_te)
        m = _metrics(y_true, preds)
        m["seed"] = seed
        m["train_time_s"] = round(elapsed, 3)
        records.append(m)
        print(f"  [regression] seed={seed:3d}  R2={m['r2']:.4f}  MSE={m['mse']:.2f}")
    return records


def run_multiout_regression(impl, seeds, ntrees=3, epochs=50):
    """Multi-output regression on synthetic data."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 6))
    y = np.column_stack(
        [
            X[:, 0] * 2 + X[:, 1] - X[:, 2] + rng.standard_normal(300) * 0.1,
            X[:, 3] - X[:, 4] * 0.5 + rng.standard_normal(300) * 0.1,
        ]
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    records = []
    for seed in seeds:
        model = _make_model(impl, ntrees)
        t0 = time.perf_counter()
        _fit(impl, model, X_tr, y_tr, epochs=epochs, seed=seed)
        elapsed = time.perf_counter() - t0

        preds = model.predict(X_te)
        m = _metrics(y_te, preds)
        m["seed"] = seed
        m["train_time_s"] = round(elapsed, 3)
        records.append(m)
        print(f"  [multiout]   seed={seed:3d}  R2={m['r2']:.4f}  MSE={m['mse']:.4f}")
    return records


def run_bma_uncertainty(impl, seeds, ntrees=3, epochs=50, n_iters=10):
    """Check that Bayesian predictions produce a well-formed uncertainty estimate.

    Uses bma_predict() on PT and bayesian_predict() on TF — both are
    normalised to the same samples array shape via _bma_samples().
    """
    X, y = load_diabetes(return_X_y=True)
    y = y.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr, X_te, y_tr, y_te, _ = scale(X_tr, X_te, y_tr, y_te)

    records = []
    for seed in seeds:
        model = _make_model(impl, ntrees)
        _fit(impl, model, X_tr, y_tr, epochs=epochs, seed=seed)

        samples = _bma_samples(model, X_te, n_iters)
        # samples shape: (n_iters * n_trees, n_test, n_out)
        std_pred = samples.std(axis=0).flatten()

        records.append(
            {
                "seed": seed,
                "mean_uncertainty": float(std_pred.mean()),
                "max_uncertainty": float(std_pred.max()),
                "bma_shape": list(samples.shape),
            }
        )
        print(f"  [bma]        seed={seed:3d}  mean_std={std_pred.mean():.4f}")
    return records


def run_hyperparams(impl, n_trials=5):
    """Check that get_hyperparameters returns sensible, stable values."""
    X, y = load_diabetes(return_X_y=True)
    y = y.reshape(-1, 1)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    sx, sy = MinMaxScaler(), MinMaxScaler()
    X_tr = sx.fit_transform(X_tr)
    y_tr = sy.fit_transform(y_tr)

    records = []
    for i in range(n_trials):
        model = djinn.DJINN_Regressor()
        params = model.get_hyperparameters(X_tr, y_tr)
        records.append(
            {
                k: (
                    int(v)
                    if isinstance(v, np.integer)
                    else float(v) if isinstance(v, np.floating) else v
                )
                for k, v in params.items()
            }
        )
        print(f"  [hyperparams] trial={i}  {params}")
    return records


def run_architecture(impl, ntrees=3, epochs=50):
    """Capture network architecture and prediction shape/dtype.

    Uses a fixed seed (0) and a small dataset so this runs quickly.
    Records:
      - predict_shape : output shape for a single test sample
      - predict_dtype : numpy dtype of predict() output
      - n_trees       : number of trees in the ensemble
      - trees         : per-tree layer shapes (PT only; TF records what it can)
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4))
    y = (X[:, 0] + X[:, 1] ** 2).reshape(-1, 1)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)

    model = _make_model(impl, ntrees)
    _fit(impl, model, X_train, y_train, epochs=epochs, seed=0)

    # prediction shape and dtype
    single_pred = model.predict(X_test[[0]])
    multi_pred = model.predict(X_test)

    record = {
        "n_trees": ntrees,
        "predict_shape_single": list(single_pred.shape),
        "predict_shape_multi": list(multi_pred.shape),
        "predict_dtype": str(single_pred.dtype),
        "n_inputs": X_train.shape[1],
        "n_outputs": y_train.shape[1],
        "trees": {},
    }

    # per-tree layer shapes
    # PT: models are nn.Module objects stored in __models dict
    pt_models = getattr(model, "_DJINN_Regressor__models", None)
    if pt_models is not None:
        for tree_idx, nn_model in pt_models.items():
            layers = []
            for name, param in nn_model.named_parameters():
                layers.append(
                    {
                        "name": name,
                        "shape": list(param.shape),
                    }
                )
            record["trees"][str(tree_idx)] = layers

    # TF: extract layer info from nninfo weights if available
    nninfo = getattr(model, "nninfo", None)
    if nninfo is not None and "final_weights" in nninfo:
        for tree_key, weight_list in nninfo["final_weights"].items():
            layers = [
                {"name": f"layer_{i}_weight", "shape": list(w.shape)}
                for i, w in enumerate(weight_list)
            ]
            record["trees"][str(tree_key)] = layers

    # Report
    print(
        f"  [architecture]  predict_shape={record['predict_shape_multi']}"
        f"  dtype={record['predict_dtype']}"
        f"  n_trees={ntrees}"
    )
    for tid, layers in record["trees"].items():
        shapes = [layer["shape"] for layer in layers]
        print(f"    tree {tid}: {shapes}")

    return record


def cleanup(keep_files=("results_tf.json", "results_pt.json")):
    """Remove model artifacts written during collection runs.

    This removes both legacy and current output patterns, including:
    - checkpoint
    - nn_info_djinn.pkl
    - djinn_model_tree*
    - djinn_model* (directories and json sidecars)

    Parameters
    ----------
    keep_files : tuple of str
        Filenames to preserve — defaults to the two results JSON files.
        Relative to the current working directory.
    """
    cwd = Path(".")
    keep = {Path(f).resolve() for f in keep_files}
    removed = []

    # Match both TF and PT output artifacts.
    patterns = [
        "checkpoint",
        "nn_info_djinn.pkl",
        "djinn_model_tree*",
        "djinn_model*",
    ]

    seen = set()
    for pattern in patterns:
        for path in sorted(cwd.glob(pattern)):
            resolved = path.resolve()
            if resolved in keep or resolved in seen:
                continue

            if path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path))
                seen.add(resolved)
                continue

            if path.is_file():
                path.unlink()
                removed.append(str(path))
                seen.add(resolved)

    if removed:
        print(f"\n  Cleaned up {len(removed)} item(s):")
        for item in removed:
            print(f"    removed: {item}")
    else:
        print("\n  Nothing to clean up.")


def main():
    """Run benchmark collection and write consolidated JSON results.

    Returns
    -------
    None
        Writes the output JSON file and prints progress.
    """
    parser = argparse.ArgumentParser(description="Collect DJINN benchmark results")
    parser.add_argument(
        "--impl",
        required=True,
        choices=["tf", "pt"],
        help="Which implementation: 'tf' or 'pt'",
    )
    parser.add_argument(
        "--out", required=True, help="Output JSON file path, e.g. results_pt.json"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="Number of random seeds to test (default: 20)",
    )
    parser.add_argument("--ntrees", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    print(f"\n=== Running {args.impl.upper()} DJINN (seeds 0-{args.seeds - 1}) ===\n")

    results = {
        "impl": args.impl,
        "ntrees": args.ntrees,
        "epochs": args.epochs,
        "regression": run_regression(args.impl, seeds, args.ntrees, args.epochs),
        "multiout": run_multiout_regression(args.impl, seeds, args.ntrees, args.epochs),
        "bma": run_bma_uncertainty(args.impl, seeds[:5], args.ntrees, args.epochs),
        "hyperparams": run_hyperparams(args.impl),
        "architecture": run_architecture(args.impl, args.ntrees, args.epochs),
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Results written to {args.out}")

    print("\n=== Cleaning up model output files ===")
    cleanup(keep_files=[args.out])


if __name__ == "__main__":
    main()
