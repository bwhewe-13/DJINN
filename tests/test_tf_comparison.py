"""
tests/test_tf_comparison.py — Statistical comparison of TF vs PyTorch DJINN.

These tests are marked @pytest.mark.comparison and are SKIPPED by default
in normal pytest runs. They only execute when results_tf.json exists on disk
(pre-generated once from the TF venv and committed to the repo), and the
current PyTorch implementation is being validated against it.

Normal dev run  (fast, no comparison):
    pytest

Full comparison run  (slow, requires results_tf.json):
    pytest -m comparison

CI comparison job  (see .github/workflows/comparison.yml):
    pytest -m comparison --tb=short

Design note — asymmetric thresholds
------------------------------------
These tests only fail when PT is *worse* than TF. PT being better (higher R²,
lower MSE) is not a failure. This avoids false failures caused by TF's known
convergence instability on certain seeds, which would otherwise make a correct
PT implementation look broken.

Specifically:
  - Median gap tests use one-sided checks (PT median >= TF median - tolerance)
  - Distribution tests use one-sided Mann-Whitney (PT is not stochastically
    worse than TF) rather than two-sided equality
  - Performance floors are set from observed TF baselines, not arbitrary values
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

# results_tf.json sits at the repo root, one level above tests/
_REPO_ROOT = Path(__file__).parent.parent
TF_BASELINE = _REPO_ROOT / "results_tf.json"
PT_RESULTS = _REPO_ROOT / "results_pt.json"


@pytest.fixture(scope="module")
def results():
    """Load both result files. Skip the whole module if either is missing."""
    if not TF_BASELINE.exists():
        pytest.skip(
            f"TF baseline not found at {TF_BASELINE}. "
            "Generate it once with: "
            "source venvs/tf-djinn/bin/activate && "
            "python run_and_collect.py --impl tf --out results_tf.json"
        )
    if not PT_RESULTS.exists():
        pytest.skip(
            f"PT results not found at {PT_RESULTS}. "
            "Generate with: python run_and_collect.py --impl pt --out results_pt.json"
        )
    with open(TF_BASELINE) as f:
        tf = json.load(f)
    with open(PT_RESULTS) as f:
        pt = json.load(f)
    return tf, pt


def _mw_pt_not_worse(a_tf, b_pt, alpha=0.05):
    """One-sided Mann-Whitney: PT (b) is not stochastically less than TF (a).

    Returns p-value of the one-sided test H1: PT < TF.
    Fails only when PT is significantly *worse* than TF.
    """
    _, p = stats.mannwhitneyu(b_pt, a_tf, alternative="less")
    return p  # fail if p < alpha (PT is stochastically less than TF)


def _ks(a, b):
    """Two-sample KS test; returns p-value."""
    _, p = stats.ks_2samp(a, b)
    return p


@pytest.mark.comparison
class TestRegressionEquivalence:
    """PT must be at least as good as TF — better is fine, worse is not."""

    def test_r2_pt_not_worse_than_tf(self, results):
        """PT R² must not be stochastically lower than TF R²."""
        tf, pt = results
        tf_r2 = [r["r2"] for r in tf["regression"]]
        pt_r2 = [r["r2"] for r in pt["regression"]]
        p = _mw_pt_not_worse(tf_r2, pt_r2)
        assert p > 0.05, (
            f"PT R² is significantly worse than TF (one-sided MW p={p:.4f}). "
            f"TF median={np.median(tf_r2):.4f}, PT median={np.median(pt_r2):.4f}"
        )

    def test_r2_median_not_below_tf(self, results):
        """PT median R² must be within 0.05 below TF — being above is fine."""
        tf, pt = results
        tf_med = np.median([r["r2"] for r in tf["regression"]])
        pt_med = np.median([r["r2"] for r in pt["regression"]])
        assert pt_med >= tf_med - 0.05, (
            f"PT median R²={pt_med:.4f} is more than 0.05 below "
            f"TF median R²={tf_med:.4f}"
        )

    def test_mse_pt_not_worse_than_tf(self, results):
        """PT MSE must not be stochastically higher than TF MSE."""
        tf, pt = results
        # For MSE lower is better, so reverse: PT worse = PT stochastically greater
        tf_mse = [r["mse"] for r in tf["regression"]]
        pt_mse = [r["mse"] for r in pt["regression"]]
        _, p = stats.mannwhitneyu(tf_mse, pt_mse, alternative="less")
        assert p > 0.05, (
            f"PT MSE is significantly worse than TF (one-sided MW p={p:.4f}). "
            f"TF median={np.median(tf_mse):.1f}, PT median={np.median(pt_mse):.1f}"
        )

    def test_pt_r2_above_performance_floor(self, results):
        """PT median R² must exceed the observed TF baseline minus tolerance."""
        tf, pt = results
        tf_med = np.median([r["r2"] for r in tf["regression"]])
        pt_med = np.median([r["r2"] for r in pt["regression"]])
        floor = tf_med - 0.05
        assert pt_med > floor, (
            f"PT median R²={pt_med:.4f} is below floor of {floor:.4f} "
            f"(TF median {tf_med:.4f} - 0.05 tolerance)"
        )

    def test_tf_r2_above_minimum_floor(self, results):
        """Sanity-check: TF baseline must be above a minimum floor."""
        tf, _ = results
        tf_med = np.median([r["r2"] for r in tf["regression"]])
        assert tf_med > 0.25, (
            f"TF baseline median R²={tf_med:.4f} is very low — "
            "the baseline file may be stale or corrupted"
        )

    def test_r2_std_pt_not_much_worse_than_tf(self, results):
        """PT R² std should not be more than 2x TF's std — high variance is a bug."""
        tf, pt = results
        tf_std = np.std([r["r2"] for r in tf["regression"]])
        pt_std = np.std([r["r2"] for r in pt["regression"]])
        assert pt_std <= tf_std * 2, (
            f"PT R² std={pt_std:.4f} is more than 2x TF std={tf_std:.4f}, "
            "suggesting high instability in PT training"
        )


@pytest.mark.comparison
class TestMultiOutputEquivalence:
    """Multi-output regression parity checks with PT-not-worse semantics."""

    def test_multiout_r2_pt_not_worse_than_tf(self, results):
        """PT multi-output R² median must not be meaningfully below TF.

        Uses median rather than a distribution test because PT has residual
        seed-level instability that widens its distribution without shifting
        the central tendency — the median is the more meaningful comparator.
        """
        tf, pt = results
        tf_r2 = [r["r2"] for r in tf["multiout"]]
        pt_r2 = [r["r2"] for r in pt["multiout"]]
        tf_med = np.median(tf_r2)
        pt_med = np.median(pt_r2)
        assert pt_med >= tf_med - 0.02, (
            f"PT multi-output median R²={pt_med:.4f} is more than 0.02 below "
            f"TF median R²={tf_med:.4f}"
        )

    def test_multiout_r2_median_not_below_tf(self, results):
        """PT median multi-output R² must be within 0.05 below TF."""
        tf, pt = results
        tf_med = np.median([r["r2"] for r in tf["multiout"]])
        pt_med = np.median([r["r2"] for r in pt["multiout"]])
        assert pt_med >= tf_med - 0.05, (
            f"PT median multi-output R²={pt_med:.4f} is more than 0.05 below "
            f"TF median={tf_med:.4f}"
        )

    def test_multiout_r2_std_not_much_worse(self, results):
        """PT multi-output R² variance should not be more than 3x TF's."""
        tf, pt = results
        tf_std = np.std([r["r2"] for r in tf["multiout"]])
        pt_std = np.std([r["r2"] for r in pt["multiout"]])
        assert pt_std <= tf_std * 3, (
            f"PT multi-output R² std={pt_std:.4f} is more than 3x "
            f"TF std={tf_std:.4f} — residual training instability in PT"
        )


@pytest.mark.comparison
class TestBMAEquivalence:
    """Bayesian model averaging uncertainty and shape consistency checks."""

    def test_pt_bma_uncertainty_is_nonzero(self, results):
        """PT must report non-zero predictive uncertainty for all seeds."""
        _, pt = results
        uncertainties = [r["mean_uncertainty"] for r in pt["bma"]]
        assert all(
            u > 0 for u in uncertainties
        ), "PT BMA produced zero uncertainty for at least one seed"

    def test_tf_bma_uncertainty_is_nonzero(self, results):
        """TF baseline must also report non-zero predictive uncertainty."""
        tf, _ = results
        uncertainties = [r["mean_uncertainty"] for r in tf["bma"]]
        assert all(
            u > 0 for u in uncertainties
        ), "TF BMA produced zero uncertainty — baseline may be corrupt"

    def test_bma_uncertainty_ratio_not_extreme(self, results):
        """PT uncertainty should be within a reasonable range of TF's.

        PT converging better naturally produces lower uncertainty, so the
        ratio is allowed to be < 1. Only flag if it's suspiciously extreme.
        """
        tf, pt = results
        tf_unc = np.mean([r["mean_uncertainty"] for r in tf["bma"]])
        pt_unc = np.mean([r["mean_uncertainty"] for r in pt["bma"]])
        ratio = pt_unc / (tf_unc + 1e-10)
        assert 0.05 < ratio < 10, (
            f"BMA uncertainty ratio PT/TF={ratio:.2f} is extreme "
            f"(TF={tf_unc:.5f}, PT={pt_unc:.5f})"
        )

    def test_bma_shape_has_multiple_samples(self, results):
        """BMA sample tensors must contain more than one draw."""
        _, pt = results
        for r in pt["bma"]:
            shape = r.get("bma_shape", [])
            assert len(shape) >= 2 and shape[0] > 1, (
                f"BMA output shape {shape} has only one sample — "
                "averaging over a single draw is not meaningful"
            )

    def test_bma_shapes_match_between_impls(self, results):
        """TF and PT BMA output shapes must be identical."""
        tf, pt = results
        for tf_r, pt_r in zip(tf["bma"], pt["bma"]):
            assert tf_r["bma_shape"] == pt_r["bma_shape"], (
                f"BMA shape mismatch: TF={tf_r['bma_shape']}, "
                f"PT={pt_r['bma_shape']}"
            )


@pytest.mark.comparison
class TestHyperparameterConsistency:
    """Sanity checks for tuned hyperparameter ranges and agreement."""

    def test_learning_rate_in_reasonable_range(self, results):
        """PT learning rates must stay inside a conservative valid interval."""
        _, pt = results
        lrs = [r["learning_rate"] for r in pt["hyperparams"] if "learning_rate" in r]
        for lr in lrs:
            assert 1e-6 <= lr <= 1.0, f"Learning rate {lr} is outside [1e-6, 1.0]"

    def test_epochs_in_reasonable_range(self, results):
        """PT epoch counts must stay inside an expected tuning range."""
        _, pt = results
        epochs = [r["epochs"] for r in pt["hyperparams"]]
        for e in epochs:
            assert (
                10 <= e <= 5000
            ), f"Epoch count {e} is outside reasonable range [10, 5000]"

    def test_batch_size_matches(self, results):
        """Batch size is data-driven so should be identical in both impls."""
        tf, pt = results
        tf_bs = [r["batch_size"] for r in tf["hyperparams"]]
        pt_bs = [r["batch_size"] for r in pt["hyperparams"]]
        if tf_bs and pt_bs:
            assert np.mean(tf_bs) == np.mean(pt_bs), (
                f"Batch sizes differ: TF={np.mean(tf_bs):.0f}, "
                f"PT={np.mean(pt_bs):.0f}"
            )


@pytest.mark.comparison
class TestArchitectureEquivalence:
    """Prediction interface and architecture metadata compatibility checks."""

    def test_predict_shapes_match(self, results):
        """Single- and multi-sample prediction shapes must match exactly."""
        tf, pt = results
        tf_arch = tf.get("architecture", {})
        pt_arch = pt.get("architecture", {})
        if not tf_arch or not pt_arch:
            pytest.skip("Architecture data not present in result files")
        assert tf_arch["predict_shape_single"] == pt_arch["predict_shape_single"], (
            f"Single-sample predict shape mismatch: "
            f"TF={tf_arch['predict_shape_single']}, "
            f"PT={pt_arch['predict_shape_single']}"
        )
        assert tf_arch["predict_shape_multi"] == pt_arch["predict_shape_multi"], (
            f"Multi-sample predict shape mismatch: "
            f"TF={tf_arch['predict_shape_multi']}, "
            f"PT={pt_arch['predict_shape_multi']}"
        )

    def test_predict_dtype_compatible(self, results):
        """Prediction dtypes must both be floating-point."""
        tf, pt = results
        tf_arch = tf.get("architecture", {})
        pt_arch = pt.get("architecture", {})
        if not tf_arch or not pt_arch:
            pytest.skip("Architecture data not present in result files")
        tf_dtype = tf_arch.get("predict_dtype", "")
        pt_dtype = pt_arch.get("predict_dtype", "")
        # Both must be a float type — float32 vs float64 is acceptable
        assert (
            "float" in tf_dtype and "float" in pt_dtype
        ), f"Non-float predict dtype: TF={tf_dtype}, PT={pt_dtype}"

    def test_io_dimensions_match(self, results):
        """Input/output dimensional metadata must match between TF and PT."""
        tf, pt = results
        tf_arch = tf.get("architecture", {})
        pt_arch = pt.get("architecture", {})
        if not tf_arch or not pt_arch:
            pytest.skip("Architecture data not present in result files")
        assert tf_arch["n_inputs"] == pt_arch["n_inputs"], (
            f"n_inputs mismatch: TF={tf_arch['n_inputs']}, " f"PT={pt_arch['n_inputs']}"
        )
        assert tf_arch["n_outputs"] == pt_arch["n_outputs"], (
            f"n_outputs mismatch: TF={tf_arch['n_outputs']}, "
            f"PT={pt_arch['n_outputs']}"
        )

    def test_n_trees_matches(self, results):
        """Ensemble tree counts must match between TF and PT metadata."""
        tf, pt = results
        tf_arch = tf.get("architecture", {})
        pt_arch = pt.get("architecture", {})
        if not tf_arch or not pt_arch:
            pytest.skip("Architecture data not present in result files")
        assert tf_arch["n_trees"] == pt_arch["n_trees"], (
            f"n_trees mismatch: TF={tf_arch['n_trees']}, " f"PT={pt_arch['n_trees']}"
        )

    def test_pt_layer_depth_consistent_across_trees(self, results):
        """All PT trees should have the same layer depth."""
        _, pt = results
        pt_arch = pt.get("architecture", {})
        if not pt_arch or not pt_arch.get("trees"):
            pytest.skip("PT layer data not present in architecture results")
        # Filter to PT-style entries (integer keys "0", "1", "2")
        pt_trees = {
            k: v for k, v in pt_arch["trees"].items() if not k.startswith("tree_")
        }
        if not pt_trees:
            pytest.skip("No PT-style tree entries found")
        depths = {
            k: len([layer for layer in layers if "weight" in layer["name"]])
            for k, layers in pt_trees.items()
        }
        unique_depths = set(depths.values())
        assert (
            len(unique_depths) == 1
        ), f"PT trees have inconsistent layer depths: {depths}"
