calculate_all_metrics
=====================

.. currentmodule:: postprocessinglib.evaluation.metrics

.. autofunction:: calculate_all_metrics

`JUPYTER NOTEBOOK TUTORIAL <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks>`_
Example
^^^^^^^
Calculation of all available metrics

.. code-block:: python
    :emphasize-lines: 1,3, 5
    :linenos:
    
    from postprocessinglib.evaluation import metrics

    observed_all, simulated_all = metrics.generate_dataframes(csvfpath="MESH_output_streamflow.csv", num_min=365)

    print(metrics.calculate_all_metrics(observed=observed_all, simulated=simulated_all, num_stations=1))

    {'MSE': [1889.8829356273197], 'RMSE': [43.47278384952268], 'MAE': [25.140806861503677], 'NSE': [0.0994826408059557],
    'NegNSE': [-0.0994826408059557], 'LogNSE': [-0.33425398964890385], 'NegLogNSE': [0.33425398964890385], 'KGE': [0.4391875106526365],
    'NegKGE': [-0.4391875106526365], 'KGE 2012': [0.3130173067471582], 'BIAS': [-34.59860016110435], 'AbsBIAS': [34.59860016110435],
    'TTP_obs': [154.8], 'TTP_sim': [180.84], 'TTCoM_obs': [184.87057921300135], 'TTCoM_sim': [190.60358617846887], 'SPOD_obs': [72.0],
    'SPOD_sim': [76.12]}