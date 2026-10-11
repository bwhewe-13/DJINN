Examples
========
Each example is a Jupyter notebook. Download it to run it yourself, or open it
in Google Colab and add ``%pip install djinnml`` as the first cell. The
notebooks take a minute or two each on a laptop CPU.

.. list-table::
   :header-rows: 1
   :widths: 30 52 18

   * - Notebook
     - What it covers
     - Get it
   * - :doc:`01_quickstart`
     - Regression on California housing: the hyperparameter search, ``fit``,
       ``predict``, ``score``, then ``save``, ``load`` and
       ``continue_training``.
     - :download:`.ipynb <01_quickstart.ipynb>`
       · `Colab <https://colab.research.google.com/github/bwhewe-13/DJINN/blob/master/docs/examples/01_quickstart.ipynb>`__
   * - :doc:`02_classification`
     - Wine cultivars with string labels: ``predict_proba``, ``classes_``
       and a confusion matrix.
     - :download:`.ipynb <02_classification.ipynb>`
       · `Colab <https://colab.research.google.com/github/bwhewe-13/DJINN/blob/master/docs/examples/02_classification.ipynb>`__
   * - :doc:`03_multioutput`
     - Predicting the range and height of a projectile together.
     - :download:`.ipynb <03_multioutput.ipynb>`
       · `Colab <https://colab.research.google.com/github/bwhewe-13/DJINN/blob/master/docs/examples/03_multioutput.ipynb>`__
   * - :doc:`04_uncertainty`
     - Bayesian DJINN with dropout: prediction intervals with ``bma_predict``
       and checking how often they cover the true values.
     - :download:`.ipynb <04_uncertainty.ipynb>`
       · `Colab <https://colab.research.google.com/github/bwhewe-13/DJINN/blob/master/docs/examples/04_uncertainty.ipynb>`__
   * - :doc:`05_scikit_learn`
     - DataFrames, ``Pipeline``, ``ColumnTransformer``, ``cross_val_score``
       against a random forest and gradient boosting, ``GridSearchCV`` and
       pickling.
     - :download:`.ipynb <05_scikit_learn.ipynb>`
       · `Colab <https://colab.research.google.com/github/bwhewe-13/DJINN/blob/master/docs/examples/05_scikit_learn.ipynb>`__
   * - :doc:`06_how_djinn_works`
     - Step by step: grow a tree, map it to a network, look at the starting
       weights, then train.
     - :download:`.ipynb <06_how_djinn_works.ipynb>`
       · `Colab <https://colab.research.google.com/github/bwhewe-13/DJINN/blob/master/docs/examples/06_how_djinn_works.ipynb>`__

.. toctree::
   :hidden:

   01_quickstart
   02_classification
   03_multioutput
   04_uncertainty
   05_scikit_learn
   06_how_djinn_works
