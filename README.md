
Deep Jointly-Informed Neural Networks
======================================
**DJINN: Deep jointly-informed neural networks**


[![Tests](https://github.com/bwhewe-13/DJINN/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/bwhewe-13/DJINN/actions/workflows/tests.yml)
[![Formatting](https://github.com/bwhewe-13/DJINN/actions/workflows/formatting.yml/badge.svg?branch=master)](https://github.com/bwhewe-13/DJINN/actions/workflows/formatting.yml)
[![codecov](https://codecov.io/gh/bwhewe-13/DJINN/branch/master/graph/badge.svg)](https://codecov.io/gh/bwhewe-13/DJINN)
[![Docs](https://github.com/bwhewe-13/DJINN/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/bwhewe-13/DJINN/actions/workflows/docs.yml)
[![License: BSD](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/bwhewe-13/DJINN/blob/master/LICENSE)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)


Fork notice: This repository is a fork and continuation of LLNL's DJINN project (Deep Jointly-Informed Neural Networks) originally developed by Kelli D. Humbird (humbird1@llnl.gov). The original project is available at https://github.com/LLNL/djinn and is distributed under the license found in `LICENSE`. This fork is maintained by Ben Whewell (ben.whewell@pm.me) — https://github.com/bwhewe-13/DJINN


DJINN is an easy-to-use algorithm for training deep neural networks on supervised regression tasks.
For additional information, refer to the paper "Deep neural network initialization with decision trees", cited below.



Getting Started
---------------
Original DJINN required TensorFlow. This fork is implemented with PyTorch.

Requirements:

- Python 3.8+
- PyTorch
- scikit-learn

Install from source:

```bash
git clone https://github.com/bwhewe-13/DJINN.git
cd DJINN
python -m pip install --upgrade pip
python -m pip install .
```

Try it out using the examples in [examples](./examples):

- `python examples/djinn_regression_example.py`
- `python examples/djinn_classification_example.py`
- `python examples/djinn_multiout_example.py`

Notes:

- The scikit-learn version used when training a DJINN model should match the
    version used when loading/evaluating that saved model.
- Some example workflows may require `matplotlib`:

    ```bash
    python -m pip install matplotlib
    ```


Development
-----------
Set up a local development environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Run quality checks and tests:

```bash
black --check djinn tests examples
isort --check-only djinn tests examples
flake8
pytest
```

Enable pre-commit hooks (optional, recommended):

```bash
pre-commit install
pre-commit run --all-files
```

Build docs locally:

```bash
python -m pip install sphinx
cd docs
make html
```

### Documentation
To view the DJINN documentation:

```
cd docs
make html
```
Open docs/_build/html/index.html in a browser


Source Repo Verification
------------------------

These tests verify that this PyTorch fork produces results consistent with
the original TensorFlow DJINN implementation.

The verification suite has two layers:

- **`compare_results.py`** — an exploratory reporting script that prints a
  human-readable comparison of two JSON result files. Useful for investigating
  differences interactively.
- **`tests/test_tf_comparison.py`** — pytest tests that formally gate the
  comparison. These use asymmetric thresholds: they only fail when PT is
  *worse* than TF, not when PT is better. This avoids false failures caused
  by TF's known convergence instability on certain seeds.

### One-time setup

Run `setup_envs.sh` from the repo root to create both virtual environments:

```bash
bash setup_envs.sh
```

This clones the TF repo into `repos/DJINN-tf/` and installs both environments:

- `venvs/tf-djinn/` — TensorFlow implementation
- `venvs/pt-djinn/` — PyTorch implementation (installed from the current repo)

### Step 1: Unit tests (run in both envs)

These tests check API compatibility, output shapes, determinism, and
save/load correctness. Run them independently in each environment:

```bash
# TensorFlow env — shared contract only
source venvs/tf-djinn/bin/activate
pytest tests/test_unit_shared.py -v
deactivate

# PyTorch env — shared contract + PT-specific behaviour
source venvs/pt-djinn/bin/activate
pytest tests/test_unit_shared.py tests/test_unit.py -v
deactivate
```

Any test that fails in one environment but passes in the other reveals
a **behavioral divergence** between the two implementations.

### Step 2: Collect benchmark results

Run the benchmark script once in each environment. This trains DJINN
across 20 random seeds and saves the metrics to JSON. The TF results
are committed to the repo as a baseline; only the PT results need to
be regenerated on each comparison run.

#### TF baseline (`results_tf.json`)

`results_tf.json` is committed to the repo root and should be treated as
stable. Only regenerate it if the TF implementation, datasets, or collection
methodology change. It was generated with:

- **TensorFlow version:** 2.21.0
- **Command:** `python run_and_collect.py --impl tf --out results_tf.json --ntrees 3 --epochs 100 --seeds 20`

To regenerate:

```bash
source venvs/tf-djinn/bin/activate
python run_and_collect.py --impl tf --out results_tf.json --ntrees 3 --epochs 100 --seeds 20
git add results_tf.json
git commit -m "Regenerate TF baseline (TF 2.21.0, ntrees=3, epochs=100, seeds=20)"
deactivate
```

```bash
# TensorFlow env — generate committed baseline (one-time)
source venvs/tf-djinn/bin/activate
python run_and_collect.py --impl tf --out results_tf.json --ntrees 3 --epochs 100
deactivate

# PyTorch env — regenerate on each comparison run
source venvs/pt-djinn/bin/activate
python run_and_collect.py --impl pt --out results_pt.json --ntrees 3 --epochs 100
deactivate
```

### Step 3: Compare results

**Exploratory report** (human-readable, no pass/fail gates):

```bash
python compare_results.py --tf results_tf.json --pt results_pt.json

# Optional: generate distribution plots (requires matplotlib)
python compare_results.py --tf results_tf.json --pt results_pt.json --plot
```

**Formal pytest comparison** (used in CI):

```bash
source venvs/pt-djinn/bin/activate
pytest tests/test_tf_comparison.py -m comparison -v
```

### Interpreting `compare_results.py` output

`compare_results.py` is an exploratory tool. Its PASS/WARN/FAIL labels use
two-sided statistical tests and are intended to guide investigation, not to
formally gate correctness. See `test_tf_comparison.py` for the authoritative
pass/fail criteria used in CI.

| Color / Status | Meaning |
|----------------|---------|
| Green `PASS`   | Distributions are not significantly different (KS + Mann-Whitney p > 0.05) |
| Yellow `WARN`  | Marginal difference — worth investigating but not necessarily a bug |
| Red `FAIL`     | Statistically significant difference or metric below performance floor |

A `FAIL` in `compare_results.py` does **not** necessarily mean the PT
implementation is wrong. PT often scores better than TF (higher R², lower
MSE, lower variance across seeds), which also triggers a FAIL under two-sided
tests. Use the pytest suite to determine whether a difference is a real
regression.

### Acceptance thresholds

These are the criteria used in `tests/test_tf_comparison.py`. All checks are
**asymmetric**: PT is only required to be at least as good as TF, not
identical to it.

| Check | Threshold |
|-------|-----------|
| Network architecture | Exact match |
| Prediction shape | Exact match |
| Prediction dtype | Must be float (float32 vs float64 acceptable) |
| Same-seed determinism | rtol=1e-4 |
| Save/load round-trip | rtol=1e-5 |
| PT median R² not below TF | PT median ≥ TF median − 0.05 |
| PT R² not stochastically worse | One-sided Mann-Whitney p > 0.05 |
| PT R² variance | PT std ≤ TF std × 2 |
| PT multi-output median R² | PT median ≥ TF median − 0.02 |
| PT multi-output variance | PT std ≤ TF std × 3 |
| BMA uncertainty > 0 | Required for all seeds |
| BMA uncertainty ratio PT/TF | Between 0.05× and 10× |
| BMA output shape | Exact match between implementations |
| Batch size | Exact match (data-driven, should be identical) |
| Learning rate | In range [1e-6, 1.0] |

### Notes

- The two implementations will **never produce identical outputs** —
  TF and PyTorch have different RNGs and optimizer defaults.
- The goal is that PT is at least as good as TF, not that they are numerically
  identical.
- PT consistently converges more reliably than TF (lower R² variance across
  seeds), particularly on multi-output tasks.
- Use at least 100 epochs for meaningful comparison results.
- `results_tf.json` is committed to the repo root and must not be deleted or
  regenerated casually — it was produced with TensorFlow 2.21.0 using
  `--ntrees 3 --epochs 100 --seeds 20`. Regenerate only when the TF
  implementation or datasets change, and update the commit message with the
  TF version and command used.
- `results_pt.json` should never be committed — add it to `.gitignore`.

---

Source Repo
-----------

DJINN is available at https://github.com/LLNL/DJINN


Citing DJINN
-----------
If you use DJINN in your research, please cite the following paper:

K. D. Humbird, J. L. Peterson and R. G. Mcclarren, "Deep Neural Network Initialization With Decision Trees," in IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 5, pp. 1286-1295, May 2019.
doi: 10.1109/TNNLS.2018.2869694,
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8478232&isnumber=8695188




Release
-----------
Copyright (c) 2018, Lawrence Livermore National Security, LLC.

Produced at the Lawrence Livermore National Laboratory

Written by K. Humbird (humbird1@llnl.gov), L. Peterson (peterson76@llnl.gov).

LLNL-CODE-754815   OCEC-18-117

All rights reserved.

Unlimited Open Source- BSD Distribution.

For release details and restrictions, please read the RELEASE, LICENSE, and NOTICE files, linked below:
- [RELEASE](./RELEASE)
- [LICENSE](./LICENSE)
- [NOTICE](./NOTICE)
