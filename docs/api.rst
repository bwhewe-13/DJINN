API reference
=============
Everything you need is in the ``djinn`` package:

.. code-block:: python

    from djinn import DJINN_Regressor, DJINN_Classifier, load

.. currentmodule:: djinn

Models
------
.. autoclass:: DJINN_Regressor
    :members:
    :inherited-members:
    :exclude-members: get_metadata_routing, set_predict_request,
                      set_predict_proba_request, set_score_request,
                      set_fit_request, nninfo, model_name,
                      model_path

.. autoclass:: DJINN_Classifier
    :members:
    :inherited-members:
    :exclude-members: get_metadata_routing, set_predict_request,
                      set_predict_proba_request, set_score_request,
                      set_fit_request, nninfo, model_name,
                      model_path

Loading
-------
.. autofunction:: load

Internals
---------
These modules do the work behind the models. They aren't needed for normal
use, but :doc:`examples/06_how_djinn_works` uses them to show each step.

djinn.random_forest
~~~~~~~~~~~~~~~~~~~
.. automodule:: djinn.random_forest
    :members:

djinn.neural_network
~~~~~~~~~~~~~~~~~~~~
.. automodule:: djinn.neural_network
    :members:
