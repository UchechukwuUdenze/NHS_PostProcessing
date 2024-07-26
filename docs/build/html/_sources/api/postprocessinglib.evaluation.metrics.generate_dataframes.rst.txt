generate_dataframes
===================

.. currentmodule:: postprocessinglib.evaluation.metrics

.. autofunction:: generate_dataframes

`JUPYTER NOTEBOOK EXAMPLE <../notebooks/Examples.ipynb>`_

Example
^^^^^^^
Generate the observed and simulated dataframes from the csv file

.. code-block:: python
    :emphasize-lines: 3,23
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

    observed, simulated = metrics.generate_dataframes(csv_fpath = filepath)
    print(observed)
              obs1      obs2
    1981  0.966878  0.053977
    1982  0.188252  0.941848
    1983  0.430902  0.963190
    1984  0.718644  0.031072
    1985  0.586581  0.541689
    1986  0.380978  0.737498
    1987  0.072452  0.188173
    1988  0.833037  0.913704
    1989  0.434239  0.425448
    1990  0.698412  0.693588

    print(simulated)
              sim1      sim2
    1981  0.348580  0.043133
    1982  0.739990  0.580866
    1983  0.292824  0.798885
    1984  0.098746  0.446317
    1985  0.479616  0.639898
    1986  0.193639  0.025509
    1987  0.095210  0.357554
    1988  0.542694  0.963027
    1989  0.817284  0.865841
    1990  0.484796  0.981778