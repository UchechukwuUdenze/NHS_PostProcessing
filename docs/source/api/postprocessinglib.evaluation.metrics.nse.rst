nse
===

.. currentmodule:: postprocessinglib.evaluation.metrics

.. autofunction:: nse

`JUPYTER NOTEBOOK EXAMPLE <../notebooks/Examples.ipynb>`_

Example
^^^^^^^
Calculate the Nash-Sutcliffe Efficiency

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
    1981  0.966878  0.348580  0.053977  0.043133
    1982  0.188252  0.739990  0.941848  0.580866
    1983  0.430902  0.292824  0.963190  0.798885
    1984  0.718644  0.098746  0.031072  0.446317
    1985  0.586581  0.479616  0.541689  0.639898
    1986  0.380978  0.193639  0.737498  0.025509
    1987  0.072452  0.095210  0.188173  0.357554
    1988  0.833037  0.542694  0.913704  0.963027
    1989  0.434239  0.817284  0.425448  0.865841
    1990  0.698412  0.484796  0.693588  0.981778

    # Generate the observed and simulated Dataframes
    obs = test_df.iloc[:, [0, 2]]
    sim = test_df.iloc[:, [1, 3]]

    # Calculate the Nash-Sutcliffe Efficiency
    nse = metrics.nse(observed = obs, simulated = sim, num_stations = 2)
    print(nse)
        [-0.9712872212067771, 0.016690558297001723]