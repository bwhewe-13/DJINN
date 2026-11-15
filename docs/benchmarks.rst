Benchmarks
==========
DJINN does two things with each tree: it uses the tree to pick the shape of a
network, and it uses the tree again to set that network's starting weights.
These benchmarks try to separate the two. The question is whether a network
trained from the tree's weights ends up better than the same network trained
from random weights.

The script is ``benchmarks/init_comparison.py``. It takes a few hours on a
desktop CPU.

Setup
-----
Each dataset is split into 5 cross-validation folds (stratified for
classification). On every fold these models are fit:

``djinn``
    DJINN as usual, with 3 trees of depth 6, trained for 300 epochs.
``random_init``
    The same networks: same layers, sparsity pattern, biases, minibatches,
    optimizer and epochs. Only the weights are different. Each one is drawn
    from a normal distribution with the Xavier-style scale DJINN uses for its
    non-zero weights.
``small_forest``
    A random forest with the same 3 trees of depth 6 that DJINN starts from.
``random_forest``, ``gradient_boosting``
    scikit-learn's :class:`~sklearn.ensemble.RandomForestRegressor` and
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor` (or the
    classifiers) with default settings.

For both networks the learning rate is chosen from 0.001, 0.003 and 0.01 on an
inner 80/20 split of the training fold, then the network is refit on the whole
fold. Scores are R² for regression and accuracy for classification, on the
held-out fold.

The datasets are California housing (a random 5000 rows), diabetes,
Friedman #1 and #3 (1000 rows each, from scikit-learn's generators), breast
cancer and digits.

Results
-------
Mean score over the 5 folds:

================  =====  ===========  ============  =============  =================
Dataset           djinn  random_init  small_forest  random_forest  gradient_boosting
================  =====  ===========  ============  =============  =================
california        0.728  0.737        0.670         0.764          0.807
diabetes          0.340  0.374        0.344         0.428          0.390
friedman1         0.951  0.945        0.759         0.847          0.919
friedman3         0.901  0.896        0.827         0.883          0.876
breast_cancer     0.967  0.972        0.940         0.961          0.970
digits            0.969  0.983        0.812         0.973          0.973
================  =====  ===========  ============  =============  =================

Tree initialization was better on 14 of the 30 folds. A paired Wilcoxon test
on the fold scores gives p = 0.79, so there is no sign of a difference either
way. The validation loss curves tell the same story. After the first few
epochs the two networks are about level, and on the classification datasets
the tree-initialized networks start with a higher loss.

Both networks are well ahead of the small forest they came from, and on the
two Friedman problems, which are smooth functions, they beat the full random
forest and gradient boosting as well. On the real datasets gradient boosting
or the random forest is usually better.

Other settings
--------------
To check that this isn't down to one choice of settings, the comparison was
repeated on four of the datasets (california, diabetes, friedman1 and
breast_cancer) with one setting changed at a time. The table counts the folds
(out of 20) where each network did better; ties are left out.

====================================  ==========  ================  =========
Setting                               djinn wins  random_init wins  Wilcoxon p
====================================  ==========  ================  =========
Default (depth 6, 300 epochs)         10          7                 0.98
Trees of depth 4                      9           11                0.85
Trees of depth 8                      9           9                 0.59
50 epochs                             13          4                 0.02
200 training rows per fold            9           11                0.64
====================================  ==========  ================  =========

The one case where tree initialization helped was the short training run. With
50 epochs it was better on all five Friedman #1 folds, by 0.02 to 0.06 in R²,
and slightly better on most of the others. With 300 epochs that lead was gone.
So the tree weights do give the network a head start, but it only matters if
training stops early. They didn't help with small training sets either.

One California fold has test inputs far outside the range of the training
data (see the notes on outliers in :doc:`usage`). Both networks score poorly
on it, and how poorly varies a lot from run to run, which is why the
California means move around between settings. ``clip_inputs=True`` would
avoid this, but it was left off here.

What this means
---------------
In these tests the gain from DJINN comes from the architecture the trees
choose, not from the starting weights. That is still useful: the depth and
width of the networks come straight from the data, with no architecture search.
But there's no reason to expect a DJINN network to beat a randomly initialized
network of the same shape once both are trained to convergence.

These are small tabular datasets and a small ensemble. The `original paper
<https://doi.org/10.1109/TNNLS.2018.2869694>`_ tested other problems, and our
results may not carry over to those. If you see something different on your
own data, please open an issue.
