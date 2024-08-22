available_metrics
=================

.. currentmodule:: postprocessinglib.evaluation.metrics

.. autofunction:: available_metrics

`JUPYTER NOTEBOOK TUTORIAL <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks>`_

Example
^^^^^^^
    View the list of available metrics

.. code-block:: python
    :emphasize-lines: 1,3
    :linenos:
    
    from postprocessinglib.evaluation import metrics

    print(metrics.available_metrics())

        ['MSE', 'RMSE', 'MAE', 'NSE', 'NegNSE', 'LogNSE', 'NegLogNSE', 'KGE', 'NegKGE',
        'KGE 2012', 'BIAS', 'AbsBIAS', 'TTP', 'TTCoM', 'SPOD']