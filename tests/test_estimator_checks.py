"""
tests/test_estimator_checks.py — scikit-learn's estimator checks.

    (pt-djinn) $ pytest tests/test_estimator_checks.py -v
"""

from sklearn.utils.estimator_checks import parametrize_with_checks

from djinn import DJINN_Classifier, DJINN_Regressor

# These checks fit on a handful of samples or on data a single split
# separates, so the random forest only grows depth-1 trees, which are too
# shallow to map to a network. Pickling and pipelines are covered in
# test_sklearn.py with realistic data.
SHALLOW_TREE = "data too small or simple to grow a tree deeper than 1"
EXPECTED_FAILURES = {
    "check_pipeline_consistency": SHALLOW_TREE,
    "check_estimators_pickle": SHALLOW_TREE,
    "check_fit2d_1feature": SHALLOW_TREE,
}
CLASSIFIER_EXPECTED_FAILURES = {
    **EXPECTED_FAILURES,
    "check_classifiers_classes": SHALLOW_TREE,
}


def expected_failures(estimator):
    """Return the checks each estimator is known to fail and why."""
    if isinstance(estimator, DJINN_Classifier):
        return CLASSIFIER_EXPECTED_FAILURES
    return EXPECTED_FAILURES


@parametrize_with_checks(
    [
        DJINN_Regressor(learning_rate=0.01, epochs=50, random_state=0),
        DJINN_Classifier(learning_rate=0.01, epochs=50, random_state=0),
    ],
    expected_failed_checks=expected_failures,
)
def test_sklearn_compatible(estimator, check):
    """Run one scikit-learn estimator check."""
    check(estimator)
