Installation
============
DJINN needs Python 3.10 to 3.14, PyTorch and scikit-learn 1.6 or newer.

From PyPI
---------
The package is called ``djinnml`` on PyPI::

    pip install djinnml

Check that it works::

    python -c "from djinn import DJINN_Regressor; print('ok')"

PyTorch is installed as a dependency. If you want a GPU build, install PyTorch
first by following the instructions at https://pytorch.org, then install
``djinnml``.

From source
-----------
::

    git clone https://github.com/bwhewe-13/DJINN.git
    cd DJINN
    pip install -e ".[dev]"

The ``dev`` extra adds pytest and the formatting tools. Run the tests with
``pytest``, or ``pytest -m "not slow"`` to skip the longer ones.

Running the examples
--------------------
The :doc:`examples <examples/index>` are Jupyter notebooks. Download them from
the examples page, or run them from a clone of the repository::

    pip install djinnml jupyter matplotlib pandas
    jupyter lab docs/examples

Each notebook can also be opened in Google Colab from the link at the top of
its page. Colab doesn't come with DJINN, so add a cell with
``%pip install djinnml`` at the top first.

Building the docs
-----------------
The ``docs`` extra installs Sphinx and the notebook support. nbsphinx also
needs `pandoc <https://pandoc.org/installing.html>`_ on your ``PATH``::

    pip install -e ".[docs]"
    python -m sphinx -M html docs docs/_build

Then open ``docs/_build/html/index.html``.

CPU and GPU
-----------
Models train on the CPU by default. Pass ``device="cuda"`` for an NVIDIA GPU,
``device="mps"`` for Apple silicon, or any :class:`torch.device`, to train and
predict on that device. ``device="auto"`` uses CUDA if it is available, then
MPS, then the CPU:

.. code-block:: python

    from djinn import DJINN_Regressor

    model = DJINN_Regressor(n_trees=5, device="cuda")

DJINN's networks are small, so a GPU only pays off once there is enough data
to keep it busy. As a rough guide, comparing two CPU threads with an RTX 2060,
on 1,000 rows the CPU was about three times faster, and on 20,000 rows the GPU
was about three times faster for training and nine times faster for the
learning rate search.
