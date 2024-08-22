mse
===

.. currentmodule:: postprocessinglib.evaluation.metrics

.. autofunction:: mse

`JUPYTER NOTEBOOK TUTORIAL <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks>`_


Example
^^^^^^^
Calculate the Mean Square Error

.. code-block:: python
    :emphasize-lines: 3,28
    :linenos:
    
    import numpy as np
    import pandas as pd
    from postprocessinglib.evaluation import metrics

    # Create your index as an array
    index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    
    # Create a test dataframe
    test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    print(test_df)
              obs1      sim1      obs2      sim2
    1981  0.869720  0.914777  0.701577  0.034410
    1982  0.126930  0.150236  0.217605  0.283580
    1983  0.082436  0.066993  0.281314  0.706240
    1984  0.865263  0.720315  0.445746  0.902906
    1985  0.042514  0.702998  0.451351  0.421407
    1986  0.400267  0.756454  0.084404  0.720665
    1987  0.352093  0.178805  0.197526  0.300795
    1988  0.154050  0.027170  0.020469  0.621782
    1989  0.153899  0.492885  0.870073  0.013124
    1990  0.255068  0.559826  0.244888  0.579176

    # Generate the observed and simulated Dataframes
    obs = test_df.iloc[:, [0, 2]]
    sim = test_df.iloc[:, [1, 3]]

    # Calculating the mean square error
    mse = metrics.mse(observed = obs, simulated = sim, num_stations = 2)
    print(mse)
        [0.08408454314573567, 0.24630978725134473]
