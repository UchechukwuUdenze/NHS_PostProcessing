station_dataframe
=================

.. currentmodule:: postprocessinglib.evaluation.metrics

.. autofunction:: station_dataframe

`JUPYTER NOTEBOOK EXAMPLE <../notebooks/Examples.ipynb>`_

Example
^^^^^^^
Retrieve the various individual stations data

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
    1981  0.405658  0.656620  0.463430  0.570444
    1982  0.166756  0.567266  0.018316  0.165674
    1983  0.831708  0.396591  0.522035  0.606923
    1984  0.896720  0.864245  0.692322  0.786928
    1985  0.631849  0.326800  0.334343  0.976953
    1986  0.236162  0.967424  0.264937  0.333279
    1987  0.547575  0.939817  0.329754  0.000407
    1988  0.195756  0.636409  0.278995  0.967959
    1989  0.006018  0.355942  0.158092  0.427026
    1990  0.543909  0.067026  0.493195  0.925726

    # Generate the observed and simulated Dataframes
    obs = test_df.iloc[:, [0, 2]]
    sim = test_df.iloc[:, [1, 3]]

    # Extract the stations
    stations = metrics.station_dataframe(observed = obs, simulated = sim, 1)
    print(stations[0])
                  obs1      sim1
        1981  0.405658  0.656620
        1982  0.166756  0.567266
        1983  0.831708  0.396591
        1984  0.896720  0.864245
        1985  0.631849  0.326800
        1986  0.236162  0.967424
        1987  0.547575  0.939817
        1988  0.195756  0.636409
        1989  0.006018  0.355942
        1990  0.543909  0.067026