How DJINN works
===============
DJINN (Deep Jointly-Informed Neural Networks) was introduced by Humbird,
Peterson and McClarren [HPM2019]_. It uses decision trees to choose the
architecture and the starting weights of a set of neural networks, then trains
those networks. This page follows the paper and describes what this
implementation does at each step. :doc:`examples/06_how_djinn_works` walks
through the same steps in code.

Why start from a tree
---------------------
Building a feed-forward network means choosing how many layers to use, how wide
each one is, and how to initialize the weights. These choices matter, and
searching over them, by hand or with Bayesian optimization, is expensive.

A decision tree fit to the same data already holds a lot of that information.
Its depth says how many levels of interactions between features are useful,
and each path from the root to a leaf says which features interact and in what
order. DJINN copies that structure into a network: each depth of the tree
becomes a layer, and the connections that follow the tree get starting weights.
The tree is only a starting point. Training is free to change every weight.

In the paper, DJINN matched the accuracy of networks tuned with Bayesian
hyperparameter optimization on a range of regression and classification
problems, at a much lower computational cost [HPM2019]_. The only settings left
to the user are the number of trees and their maximum depth.

1. Grow a random forest
-----------------------
The inputs, and for regression the outputs, are scaled to [0, 1] with
:class:`~sklearn.preprocessing.MinMaxScaler`. A
:class:`~sklearn.ensemble.RandomForestRegressor` with ``n_trees`` trees and
``max_depth=max_tree_depth`` is then fit to the scaled data. Each tree is grown
on a bootstrap sample. For classification the forest is fit to the class
indices, as in the original DJINN.

Each tree becomes one network, so ``n_trees`` is also the size of the ensemble.

2. Map each tree to a network
-----------------------------
Let a tree have depth :math:`D` and :math:`s_d` split (non-leaf) nodes at depth
:math:`d`, with the root at depth 0. For :math:`p` input features and :math:`q`
outputs, the network has :math:`D + 1` layers with widths

.. math::

    n_0 = p, \qquad
    n_d = n_{d-1} + s_d \quad (1 \le d \le D - 1), \qquad
    n_D = q.

So every hidden layer keeps the neurons of the layer before it and adds one new
neuron for each split at that depth. There are :math:`D - 1` hidden layers,
which is ``max_tree_depth - 1`` when the tree grows to its full depth. A tree
of depth 1 or less gives no hidden layers and can't be used.

The weights start at zero except along the paths of the tree:

- **Splits.** The neuron for a split at depth :math:`d` is connected to the
  neuron for its parent split, or to the parent's input feature for the
  children of the root, and to the input feature the split itself uses.
- **Inputs.** An input feature is passed on to the next layer with a weight of
  1 for as long as a deeper split uses it.
- **Leaves.** Where a branch ends in a leaf, the parent's neuron is passed
  through to the last hidden layer with weights of 1 and connected to every
  output.
- **Output.** The output layer gets starting weights from the active neurons
  of the last hidden layer.

The non-zero weights that aren't pass-throughs are drawn from
:math:`\mathcal{N}\left(0, \sqrt{3 / (n_\text{in} + n_\text{out})}\right)`, a
Xavier-style scale. Biases are drawn from a normal distribution with standard
deviation :math:`\sqrt{3 / (q + n_{D-1})}`.

The result is a sparse network whose shape follows the tree. Since every tree
in the forest is different, every network in the ensemble has its own
architecture.

3. Train the networks
---------------------
Each network is trained on its own, with:

- ReLU activations on the hidden layers,
- Adam, with ``weight_decay`` as an L2 penalty,
- mean squared error for regression and cross-entropy for classification,
- minibatches of ``batch_size`` rows drawn with replacement, and
  ``n_samples // batch_size`` batches per epoch, as in the original
  TensorFlow code.

Each network trains on 90% of the training data. The other 10%, the same rows
for every network, is held back and its loss is recorded every epoch in
``nninfo_["valid_cost"]``. It's there to check for overfitting and isn't used
to stop training early.

When ``dropout_keep_prob`` is below 1, dropout is applied after the second and
later hidden layers.

4. Pick the hyperparameters
---------------------------
When ``learning_rate`` is ``None``, :meth:`~djinn.DJINN_Regressor.fit` runs
:meth:`~djinn.DJINN_Regressor.get_hyperparameters` first, using the network
from the first tree:

- **Batch size** is 5% of the training samples.
- **Learning rate.** Ten learning rates spaced evenly in log scale between
  :math:`10^{-4}` and :math:`10^{-2}` are each trained for 100 epochs. The
  range is narrowed to the two with the lowest loss over the last 10 epochs,
  and the search is repeated once on that range.
- **Epochs.** The network is trained with that learning rate in blocks of 10
  epochs, starting after 200, until the mean loss of each of the last two
  blocks differs by less than 5% from the block before. The cap is 3000
  epochs.

That is around twenty short training runs, so for large data or repeated fits
(such as cross-validation) it pays to run the search once and pass
``learning_rate``, ``epochs`` and ``batch_size`` to the constructor.

5. Predict
----------
:meth:`~djinn.DJINN_Regressor.predict` runs every network once with dropout
off and averages their outputs. For classification, it takes the median of
the networks' softmax outputs for each class and returns the class with the
highest value.

Uncertainty with Bayesian DJINN
-------------------------------
With dropout left on at prediction time, every pass through a network uses a
different random subset of neurons. Gal and Ghahramani [GG2016]_ showed that
averaging over these passes approximates Bayesian inference over the weights,
a method known as Monte Carlo dropout.
:meth:`~djinn.DJINN_Regressor.bayesian_predict` and
:meth:`~djinn.DJINN_Regressor.bma_predict` run ``n_iters`` passes through each
network and report the 25th, 50th and 75th percentiles of the
``n_iters * n_trees`` predictions. Combining the networks this way is a form of
Bayesian model averaging.

The original DJINN examples suggest ``dropout_keep_prob`` around 0.95 with 5
to 10 trees. The spread reflects how unsure the networks are about their weights. It doesn't
include the noise in the data, so check how often the true values fall inside
the intervals on held-out data before relying on them.
:doc:`examples/04_uncertainty` shows how.

Application: neutron transport
------------------------------
Whewell and McClarren [WM2022]_ used DJINN to cut the data needed for
multigroup discrete ordinates (S\ :sub:`N`) neutron transport calculations.
These calculations store large scattering and fission cross-section matrices,
and the storage grows quickly with the number of energy groups. Instead of
storing the matrices, they trained autoencoders and DJINN models to map the
scalar flux to the scattering and fission sources. For a 618-group problem,
this reduced data storage by about 94%, while preserving the scalar flux and
lowering wall-clock time.

It's a good example of where DJINN fits: a regression problem on structured
numerical data, where a network has to be built without a long architecture
search.

References
----------
.. [HPM2019] K. D. Humbird, J. L. Peterson and R. G. McClarren, "Deep Neural
   Network Initialization With Decision Trees," *IEEE Transactions on Neural
   Networks and Learning Systems*, 30(5):1286-1295, 2019.
   `doi:10.1109/TNNLS.2018.2869694 <https://doi.org/10.1109/TNNLS.2018.2869694>`_,
   `arXiv:1707.00784 <https://arxiv.org/abs/1707.00784>`_.

.. [WM2022] B. Whewell and R. G. McClarren, "Data reduction in deterministic
   neutron transport calculations using machine learning," *Annals of Nuclear
   Energy*, 176:109276, 2022.
   `doi:10.1016/j.anucene.2022.109276 <https://doi.org/10.1016/j.anucene.2022.109276>`_,
   `arXiv:2205.05165 <https://arxiv.org/abs/2205.05165>`_.

.. [GG2016] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation:
   Representing Model Uncertainty in Deep Learning," *Proceedings of the 33rd
   International Conference on Machine Learning*, PMLR 48:1050-1059, 2016.
   `arXiv:1506.02142 <https://arxiv.org/abs/1506.02142>`_.
