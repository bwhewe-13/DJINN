"""
compare_results.py — Statistical comparison of TF vs PyTorch DJINN results.

Run this AFTER generating both JSON files:
    python run_and_collect.py --impl tf --out results_tf.json
    python run_and_collect.py --impl pt --out results_pt.json

Then:
    python compare_results.py --tf results_tf.json --pt results_pt.json

This script does NOT require either framework — only numpy, scipy, and
matplotlib (optional for plots).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def _supports_color():
    """Return True when stdout likely supports ANSI color output."""
    if os.getenv("NO_COLOR"):
        return False
    term = os.getenv("TERM", "")
    return sys.stdout.isatty() and bool(term) and term.lower() != "dumb"


def _colorize(text, code):
    """Wrap text in ANSI color codes when color output is enabled.

    Parameters
    ----------
    text : str
        Plain text to colorize.
    code : str
        ANSI color code string.

    Returns
    -------
    str
        Colorized text when ANSI output is supported, otherwise the original
        text.
    """
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


PASS = _colorize("PASS", "32")
WARN = _colorize("WARN", "33")
FAIL = _colorize("FAIL", "31")


def section(title):
    """Print a formatted section header.

    Parameters
    ----------
    title : str
        Section title text.

    Returns
    -------
    None
        This function prints output and returns nothing.
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def row(label, tf_val, pt_val, result, note=""):
    """Print one formatted comparison row.

    Parameters
    ----------
    label : str
        Name of the compared metric.
    tf_val : str
        TensorFlow value to display.
    pt_val : str
        PyTorch value to display.
    result : str
        Status marker (for example PASS/WARN/FAIL).
    note : str, optional
        Optional note appended to the row.

    Returns
    -------
    None
        This function prints output and returns nothing.
    """
    print(f"  {label:<28} TF={tf_val:>10}  PT={pt_val:>10}  {result}  {note}")


# statistical helpers
def ks_test(a, b, alpha=0.05):
    """Two-sample KS test. Returns (statistic, p_value, passed)."""
    stat, p = stats.ks_2samp(a, b)
    return stat, p, p > alpha


def mannwhitney(a, b, alpha=0.05):
    """Mann-Whitney U test. Returns (statistic, p_value, passed)."""
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return stat, p, p > alpha


def summary_stats(values):
    """Return summary statistics for a numeric sequence.

    Parameters
    ----------
    values : Sequence[float]
        Numeric values to summarize.

    Returns
    -------
    dict
        Dictionary containing ``mean``, ``std``, ``min``, ``max``, and
        ``median``.
    """
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 5),
        "std": round(float(arr.std()), 5),
        "min": round(float(arr.min()), 5),
        "max": round(float(arr.max()), 5),
        "median": round(float(np.median(arr)), 5),
    }


# comparison checks
def compare_regression(tf_data, pt_data):
    """Compare single-output regression metrics between TF and PyTorch.

    Parameters
    ----------
    tf_data : list[dict]
        TensorFlow regression result records.
    pt_data : list[dict]
        PyTorch regression result records.

    Returns
    -------
    None
        Prints a formatted comparison report.
    """
    section("1. Single-Output Regression (Diabetes Dataset)")

    for metric in ("r2", "mse", "mae"):
        tf_vals = [r[metric] for r in tf_data]
        pt_vals = [r[metric] for r in pt_data]

        tf_s = summary_stats(tf_vals)
        pt_s = summary_stats(pt_vals)

        _, p_ks, passed_ks = ks_test(tf_vals, pt_vals)
        _, p_mw, passed_mw = mannwhitney(tf_vals, pt_vals)

        # Note: two-sided tests are used here for exploratory reporting.
        # A FAIL does not necessarily mean PT is wrong — PT scoring better
        # than TF also triggers a FAIL. See tests/test_tf_comparison.py for
        # the asymmetric one-sided tests used as formal CI gates.
        overall = PASS if (passed_ks and passed_mw) else FAIL
        print(f"\n  [{metric.upper()}]")
        print(
            f"    TF  mean={tf_s['mean']:>9}  std={tf_s['std']:>9}  "
            f"min={tf_s['min']:>9}  max={tf_s['max']:>9}"
        )
        print(
            f"    PT  mean={pt_s['mean']:>9}  std={pt_s['std']:>9}  "
            f"min={pt_s['min']:>9}  max={pt_s['max']:>9}"
        )
        print(f"    KS  p={p_ks:.4f}  MW  p={p_mw:.4f}  {overall}")

    # Performance floor: median R2 must be above 0.25 (conservative floor
    # accounting for TF instability; see test_tf_comparison.py for the
    # asymmetric pytest thresholds used in CI)
    tf_r2_med = np.median([r["r2"] for r in tf_data])
    pt_r2_med = np.median([r["r2"] for r in pt_data])
    print("\n  Performance floor (median R2 > 0.25):")
    row("TF median R2", f"{tf_r2_med:.4f}", "—", PASS if tf_r2_med > 0.25 else FAIL)
    row("PT median R2", "—", f"{pt_r2_med:.4f}", PASS if pt_r2_med > 0.25 else FAIL)

    # Check mean R2 within ±0.05 of each other
    diff = abs(tf_r2_med - pt_r2_med)
    result = PASS if diff <= 0.05 else (WARN if diff <= 0.10 else FAIL)
    row(
        "R2 median gap",
        f"{tf_r2_med:.4f}",
        f"{pt_r2_med:.4f}",
        result,
        note=f"(|diff|={diff:.4f}, threshold=0.05)",
    )


def compare_multiout(tf_data, pt_data):
    """Compare multi-output regression metrics between TF and PyTorch.

    Parameters
    ----------
    tf_data : list[dict]
        TensorFlow multi-output result records.
    pt_data : list[dict]
        PyTorch multi-output result records.

    Returns
    -------
    None
        Prints a formatted comparison report.
    """
    section("2. Multi-Output Regression (Synthetic Dataset)")

    for metric in ("r2", "mse"):
        tf_vals = [r[metric] for r in tf_data]
        pt_vals = [r[metric] for r in pt_data]
        _, p_ks, passed_ks = ks_test(tf_vals, pt_vals)
        _, p_mw, passed_mw = mannwhitney(tf_vals, pt_vals)
        overall = PASS if (passed_ks and passed_mw) else FAIL
        tf_s = summary_stats(tf_vals)
        pt_s = summary_stats(pt_vals)
        print(f"\n  [{metric.upper()}]")
        print(f"    TF  mean={tf_s['mean']:>9}  std={tf_s['std']:>9}")
        print(f"    PT  mean={pt_s['mean']:>9}  std={pt_s['std']:>9}")
        print(f"    KS  p={p_ks:.4f}  MW  p={p_mw:.4f}  {overall}")


def compare_bma(tf_data, pt_data):
    """Compare Bayesian model averaging uncertainty behavior.

    Parameters
    ----------
    tf_data : list[dict]
        TensorFlow BMA result records.
    pt_data : list[dict]
        PyTorch BMA result records.

    Returns
    -------
    None
        Prints a formatted comparison report.
    """
    section("3. BMA Uncertainty Estimates")

    tf_unc = [r["mean_uncertainty"] for r in tf_data]
    pt_unc = [r["mean_uncertainty"] for r in pt_data]

    # BMA samples must be non-zero (uncertainty should exist)
    tf_nonzero = all(u > 0 for u in tf_unc)
    pt_nonzero = all(u > 0 for u in pt_unc)
    row(
        "TF BMA uncertainty > 0",
        f"{np.mean(tf_unc):.5f}",
        "—",
        PASS if tf_nonzero else FAIL,
    )
    row(
        "PT BMA uncertainty > 0",
        "—",
        f"{np.mean(pt_unc):.5f}",
        PASS if pt_nonzero else FAIL,
    )

    # BMA sample shapes must be non-trivial
    for impl, data in [("TF", tf_data), ("PT", pt_data)]:
        for r in data:
            shape = r.get("bma_shape", [])
            ok = len(shape) >= 2 and all(s > 0 for s in shape)
            print(f"  {impl} BMA shape={shape}  {PASS if ok else FAIL}")

    # Compare uncertainty magnitudes
    if tf_unc and pt_unc:
        _, p_mw, passed = mannwhitney(tf_unc, pt_unc)
        row(
            "Uncertainty distribution",
            f"mean={np.mean(tf_unc):.5f}",
            f"mean={np.mean(pt_unc):.5f}",
            PASS if passed else WARN,
            note=f"(MW p={p_mw:.4f})",
        )


def compare_hyperparams(tf_data, pt_data):
    """Compare selected hyperparameter distributions across implementations.

    Parameters
    ----------
    tf_data : list[dict]
        TensorFlow hyperparameter records.
    pt_data : list[dict]
        PyTorch hyperparameter records.

    Returns
    -------
    None
        Prints a formatted comparison report.
    """
    section("4. Hyperparameter Values")

    keys = set(tf_data[0].keys()) & set(pt_data[0].keys())
    for key in sorted(keys):
        tf_vals = [r[key] for r in tf_data if key in r]
        pt_vals = [r[key] for r in pt_data if key in r]
        try:
            tf_arr = np.array(tf_vals, dtype=float)
            pt_arr = np.array(pt_vals, dtype=float)
            tf_mean = tf_arr.mean()
            pt_mean = pt_arr.mean()
            diff = abs(tf_mean - pt_mean)
            # Allow up to 20% relative difference
            threshold = max(abs(tf_mean), abs(pt_mean)) * 0.20 + 1e-8
            result = PASS if diff <= threshold else WARN
            row(
                key,
                f"{tf_mean:.4g}",
                f"{pt_mean:.4g}",
                result,
                note=f"(|diff|={diff:.4g})",
            )
        except (TypeError, ValueError):
            # Non-numeric hyperparameter: just check they match
            match = tf_vals == pt_vals
            row(key, str(tf_vals[0]), str(pt_vals[0]), PASS if match else WARN)


def compare_architecture(tf_arch, pt_arch):
    """Compare architecture metadata and prediction interface details.

    Parameters
    ----------
    tf_arch : dict
        TensorFlow architecture summary.
    pt_arch : dict
        PyTorch architecture summary.

    Returns
    -------
    None
        Prints a formatted comparison report.
    """
    section("5. Network Architecture & Prediction Shape/Dtype")

    # prediction shape
    tf_single = tf_arch.get("predict_shape_single", [])
    pt_single = pt_arch.get("predict_shape_single", [])
    match_single = tf_single == pt_single
    row(
        "predict shape (1 sample)",
        str(tf_single),
        str(pt_single),
        PASS if match_single else FAIL,
    )

    tf_multi = tf_arch.get("predict_shape_multi", [])
    pt_multi = pt_arch.get("predict_shape_multi", [])
    match_multi = tf_multi == pt_multi
    row(
        "predict shape (N samples)",
        str(tf_multi),
        str(pt_multi),
        PASS if match_multi else FAIL,
    )

    # prediction dtype
    tf_dtype = tf_arch.get("predict_dtype", "unknown")
    pt_dtype = pt_arch.get("predict_dtype", "unknown")
    match_dtype = tf_dtype == pt_dtype
    row(
        "predict dtype",
        tf_dtype,
        pt_dtype,
        PASS if match_dtype else WARN,
        note="(float32 vs float64 is acceptable)" if not match_dtype else "",
    )

    # I/O dimensions
    tf_nin = tf_arch.get("n_inputs", "?")
    pt_nin = pt_arch.get("n_inputs", "?")
    tf_nout = tf_arch.get("n_outputs", "?")
    pt_nout = pt_arch.get("n_outputs", "?")
    row("n_inputs", str(tf_nin), str(pt_nin), PASS if tf_nin == pt_nin else FAIL)
    row("n_outputs", str(tf_nout), str(pt_nout), PASS if tf_nout == pt_nout else FAIL)

    # n_trees
    tf_ntrees = tf_arch.get("n_trees", "?")
    pt_ntrees = pt_arch.get("n_trees", "?")
    row(
        "n_trees",
        str(tf_ntrees),
        str(pt_ntrees),
        PASS if tf_ntrees == pt_ntrees else FAIL,
    )

    # per-tree layer shapes
    tf_trees = tf_arch.get("trees", {})
    pt_trees = pt_arch.get("trees", {})

    if not tf_trees or not pt_trees:
        print("\n  [layer shapes] Insufficient data in one or both results files.")
        return

    print("\n  Per-tree layer shapes:")
    all_match = True
    for tree_id in sorted(pt_trees.keys()):
        pt_layers = pt_trees.get(tree_id, [])
        tf_layers = tf_trees.get(tree_id, [])

        pt_shapes = [layer["shape"] for layer in pt_layers]
        tf_shapes = [layer["shape"] for layer in tf_layers]

        # For PT, weights are stored (out, in) due to nn.Linear convention;
        # for TF they may be (in, out). Normalise by sorting each pair so
        # we compare the set of dimensions rather than orientation.
        # pt_norm = [
        #     sorted(s)
        #     for s in pt_shapes
        #     if "weight" in pt_layers[pt_shapes.index(s)].get("name", "weight")
        # ]
        # tf_norm = [sorted(s) for s in tf_shapes]

        # Depth check: both trees should have the same number of layers
        pt_depth = len(
            [layer for layer in pt_layers if "weight" in layer.get("name", "")]
        )
        tf_depth = len(tf_layers)
        depth_match = pt_depth == tf_depth

        result = PASS if depth_match else FAIL
        if not depth_match:
            all_match = False

        print(
            f"    tree {tree_id}:  depth  TF={tf_depth}  PT={pt_depth}  " f"  {result}"
        )
        print(f"      TF shapes: {tf_shapes}")
        print(f"      PT shapes: {pt_shapes}")

    overall_str = PASS if all_match else FAIL
    print(f"\n  Overall layer depth match: {overall_str}")


def compare_training_time(tf_data, pt_data):
    """Report training-time differences between implementations.

    Parameters
    ----------
    tf_data : list[dict]
        TensorFlow regression result records.
    pt_data : list[dict]
        PyTorch regression result records.

    Returns
    -------
    None
        Prints a timing summary.
    """
    section("6. Training Speed")

    tf_times = [r["train_time_s"] for r in tf_data]
    pt_times = [r["train_time_s"] for r in pt_data]
    tf_mean = np.mean(tf_times)
    pt_mean = np.mean(pt_times)
    ratio = pt_mean / tf_mean if tf_mean > 0 else float("inf")

    print(f"  TF mean train time:  {tf_mean:.3f}s")
    print(f"  PT mean train time:  {pt_mean:.3f}s")
    print(
        f"  PT/TF ratio:         {ratio:.2f}x  "
        f"({'faster' if ratio < 1 else 'slower'})"
    )
    # No pass/fail — just informational


def main():
    """Parse CLI arguments, load result files, and run all comparisons.

    Returns
    -------
    None
        Exits the process with an error message when required files are
        missing.
    """
    parser = argparse.ArgumentParser(description="Compare TF vs PT DJINN results")
    parser.add_argument("--tf", required=True, help="Path to results_tf.json")
    parser.add_argument("--pt", required=True, help="Path to results_pt.json")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots (requires matplotlib)",
    )
    args = parser.parse_args()

    tf_path = Path(args.tf)
    pt_path = Path(args.pt)

    if not tf_path.exists():
        sys.exit(f"TF results file not found: {tf_path}")
    if not pt_path.exists():
        sys.exit(f"PT results file not found: {pt_path}")

    with open(tf_path) as f:
        tf_results = json.load(f)
    with open(pt_path) as f:
        pt_results = json.load(f)

    print(f"\n{'#'*60}")
    print("  DJINN TF vs PyTorch Comparison Report")
    print(f"  TF file: {tf_path}")
    print(f"  PT file: {pt_path}")
    print(f"  Seeds:   {tf_results['epochs']} epochs, " f"{tf_results['ntrees']} trees")
    print(f"{'#'*60}")

    compare_regression(tf_results["regression"], pt_results["regression"])
    compare_multiout(tf_results["multiout"], pt_results["multiout"])
    compare_bma(tf_results["bma"], pt_results["bma"])
    compare_hyperparams(tf_results["hyperparams"], pt_results["hyperparams"])
    if "architecture" in tf_results and "architecture" in pt_results:
        compare_architecture(tf_results["architecture"], pt_results["architecture"])
    else:
        print(
            "\n  [architecture] Not present in one or both result files — "
            "rerun run_and_collect.py to generate."
        )
    compare_training_time(tf_results["regression"], pt_results["regression"])

    if args.plot:
        _make_plots(tf_results, pt_results)

    print(f"\n{'='*60}")
    print(f"  Done. Review any {FAIL} or {WARN} items above.")
    print(f"{'='*60}\n")


def _make_plots(tf_results, pt_results):
    """Generate side-by-side distribution plots for core regression metrics.

    Parameters
    ----------
    tf_results : dict
        TensorFlow benchmark results dictionary.
    pt_results : dict
        PyTorch benchmark results dictionary.

    Returns
    -------
    None
        Saves a PNG image when matplotlib is available.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plots skipped: matplotlib not installed]")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("TF vs PyTorch DJINN — Distribution Comparison", fontsize=13)

    metrics = [("r2", "R2"), ("mse", "MSE"), ("mae", "MAE")]
    for ax, (key, label) in zip(axes, metrics):
        tf_vals = [r[key] for r in tf_results["regression"]]
        pt_vals = [r[key] for r in pt_results["regression"]]
        ax.hist(tf_vals, alpha=0.6, bins=10, label="TensorFlow", color="steelblue")
        ax.hist(pt_vals, alpha=0.6, bins=10, label="PyTorch", color="tomato")
        ax.set_title(label)
        ax.legend()
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    plt.tight_layout()
    out = "djinn_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"\n  [plot saved to {out}]")


if __name__ == "__main__":
    main()
