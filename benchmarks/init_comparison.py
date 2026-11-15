"""
Compare tree initialization with random initialization of the same networks.

For each dataset and cross-validation fold this fits:

- ``djinn``: DJINN as usual, networks initialized from the trees.
- ``random_init``: the same networks (same layer sizes, biases, data loader,
  optimizer and epochs), but every weight is drawn at random with the Xavier
  scale DJINN uses for its non-zero weights. Only the tree structure differs.
- ``small_forest``: a random forest with the same number of trees and depth
  as the one DJINN starts from.
- ``random_forest`` and ``gradient_boosting``: scikit-learn defaults.

The learning rate of both network models is picked from a small grid on an
inner validation split of each training fold.

    python benchmarks/init_comparison.py --out benchmarks/results
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    make_friedman1,
    make_friedman3,
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

LEARNING_RATES = [1e-3, 3e-3, 1e-2]
N_FOLDS = 5


def load(name):
    """Return ``(X, y, regression)`` for a benchmark dataset."""
    if name == "california":
        X, y = fetch_california_housing(return_X_y=True)
        idx = np.random.default_rng(0).choice(len(y), 5000, replace=False)
        return X[idx], y[idx], True
    if name == "diabetes":
        return (*load_diabetes(return_X_y=True), True)
    if name == "friedman1":
        return (*make_friedman1(1000, noise=1.0, random_state=0), True)
    if name == "friedman3":
        X, y = make_friedman3(1000, noise=0.1, random_state=0)
        return X, y, True
    if name == "breast_cancer":
        return (*load_breast_cancer(return_X_y=True), False)
    if name == "digits":
        return (*load_digits(return_X_y=True), False)
    raise ValueError(name)


def random_weights(build):
    """Wrap ``build_tree_weights_and_biases`` to replace the tree weights."""

    def wrapped(ttn, key, seed=None, tree_idx=0):
        weights, biases = build(ttn, key, seed=seed, tree_idx=tree_idx)
        rng = np.random.default_rng(None if seed is None else seed * 1000 + tree_idx)
        for name, w in weights.items():
            nin, nout = w.shape
            scale = np.sqrt(3.0 / (nin + nout))
            weights[name] = (rng.standard_normal(w.shape) * scale).astype(np.float32)
        return weights, biases

    return wrapped


def fit_network(X, y, regression, lr, random_init, seed, config):
    """Fit DJINN, optionally with random weights in place of the tree ones."""
    from djinn import DJINN_Classifier, DJINN_Regressor, neural_network

    cls = DJINN_Regressor if regression else DJINN_Classifier
    model = cls(
        n_trees=config["n_trees"],
        max_tree_depth=config["depth"],
        learning_rate=lr,
        epochs=config["epochs"],
        random_state=seed,
    )
    if not random_init:
        return model.fit(X, y)
    build = neural_network.build_tree_weights_and_biases
    with mock.patch.object(
        neural_network, "build_tree_weights_and_biases", random_weights(build)
    ):
        return model.fit(X, y)


def score(regression, y, pred):
    """R² for regression, accuracy for classification."""
    return r2_score(y, pred) if regression else accuracy_score(y, pred)


def tune_and_fit(X, y, regression, random_init, seed, config):
    """Pick the learning rate on an inner split, then refit on all of X."""
    X_in, X_val, y_in, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=None if regression else y
    )
    scores = []
    for lr in LEARNING_RATES:
        model = fit_network(X_in, y_in, regression, lr, random_init, seed, config)
        scores.append(score(regression, y_val, model.predict(X_val)))
    lr = LEARNING_RATES[int(np.argmax(scores))]
    return fit_network(X, y, regression, lr, random_init, seed, config), lr


def run_fold(args):
    """Fit every model on one fold and return the scores."""
    name, fold, train, test, config = args
    import torch

    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "2")))
    X, y, regression = load(name)
    if config["max_train"] and len(train) > config["max_train"]:
        rng = np.random.default_rng(fold)
        train = rng.choice(train, config["max_train"], replace=False)
    X_tr, X_te, y_tr, y_te = X[train], X[test], y[train], y[test]
    n_trees, depth = config["n_trees"], config["depth"]
    rows = []

    for model_name, random_init in [("djinn", False), ("random_init", True)]:
        model, lr = tune_and_fit(X_tr, y_tr, regression, random_init, fold, config)
        valid = np.asarray(model.nninfo_["valid_cost"]).reshape(n_trees, -1)
        rows.append(
            {
                "model": model_name,
                "score": score(regression, y_te, model.predict(X_te)),
                "learning_rate": lr,
                "valid_curve": valid.mean(axis=0).tolist(),
            }
        )

    if regression:
        small = RandomForestRegressor(n_trees, max_depth=depth, random_state=fold)
        others = [
            ("random_forest", RandomForestRegressor(random_state=fold, n_jobs=1)),
            ("gradient_boosting", HistGradientBoostingRegressor(random_state=fold)),
        ]
    else:
        small = RandomForestClassifier(n_trees, max_depth=depth, random_state=fold)
        others = [
            ("random_forest", RandomForestClassifier(random_state=fold, n_jobs=1)),
            ("gradient_boosting", HistGradientBoostingClassifier(random_state=fold)),
        ]
    for model_name, model in [("small_forest", small), *others]:
        model.fit(X_tr, y_tr)
        rows.append(
            {"model": model_name, "score": score(regression, y_te, model.predict(X_te))}
        )

    for row in rows:
        row.update(
            dataset=name, fold=fold, metric="r2" if regression else "accuracy", **config
        )
    print(name, fold, {r["model"]: round(r["score"], 3) for r in rows}, flush=True)
    return rows


def main():
    """Run every dataset and fold and save the rows as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default="benchmarks/results")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-trees", type=int, default=3)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--max-train", type=int, default=0, help="subsample each training fold"
    )
    parser.add_argument("--name", default="init_comparison")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "california",
            "diabetes",
            "friedman1",
            "friedman3",
            "breast_cancer",
            "digits",
        ],
    )
    args = parser.parse_args()

    config = {
        "n_trees": args.n_trees,
        "depth": args.depth,
        "epochs": args.epochs,
        "max_train": args.max_train,
    }
    jobs = []
    for name in args.datasets:
        X, y, regression = load(name)
        cv = (
            KFold(N_FOLDS, shuffle=True, random_state=0)
            if regression
            else StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
        )
        for fold, (train, test) in enumerate(cv.split(X, y)):
            jobs.append((name, fold, train, test, config))

    with ProcessPoolExecutor(args.workers) as pool:
        rows = [row for result in pool.map(run_fold, jobs) for row in result]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.name}.json", "w") as f:
        json.dump(rows, f)


if __name__ == "__main__":
    main()
