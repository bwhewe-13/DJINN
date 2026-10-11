Citing DJINN
============
If you use DJINN in your research, please cite the paper that introduced it:

    K. D. Humbird, J. L. Peterson and R. G. McClarren, "Deep Neural Network
    Initialization With Decision Trees," *IEEE Transactions on Neural Networks
    and Learning Systems*, 30(5):1286-1295, 2019.
    `doi:10.1109/TNNLS.2018.2869694 <https://doi.org/10.1109/TNNLS.2018.2869694>`_

.. code-block:: bibtex

    @article{humbird2019djinn,
      author  = {Humbird, Kelli D. and Peterson, J. Luc and McClarren, Ryan G.},
      title   = {Deep Neural Network Initialization With Decision Trees},
      journal = {IEEE Transactions on Neural Networks and Learning Systems},
      volume  = {30},
      number  = {5},
      pages   = {1286--1295},
      year    = {2019},
      doi     = {10.1109/TNNLS.2018.2869694}
    }

To cite this PyTorch implementation as well, use the Zenodo record. The DOI
below always points to the latest version, and the Zenodo page lists a DOI for
each release.

.. code-block:: bibtex

    @software{djinn_pytorch,
      author    = {Humbird, Kelli D. and Peterson, J. Luc and Whewell, Ben},
      title     = {{DJINN: Deep Jointly-Informed Neural Networks (PyTorch)}},
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.23158769},
      url       = {https://doi.org/10.5281/zenodo.23158769}
    }

The same information is in
`CITATION.cff <https://github.com/bwhewe-13/DJINN/blob/master/CITATION.cff>`_,
and GitHub's "Cite this repository" button exports it.

Publications using DJINN
------------------------
- B. Whewell and R. G. McClarren, "Data reduction in deterministic neutron
  transport calculations using machine learning," *Annals of Nuclear Energy*,
  176:109276, 2022.
  `doi:10.1016/j.anucene.2022.109276 <https://doi.org/10.1016/j.anucene.2022.109276>`_,
  `arXiv:2205.05165 <https://arxiv.org/abs/2205.05165>`_.
  Uses autoencoders and DJINN in place of the cross-section matrices in
  multigroup S\ :sub:`N` calculations, cutting data storage by about 94% for a
  618-group problem.
