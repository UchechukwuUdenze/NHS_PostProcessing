generate_dataframes
===================

.. currentmodule:: postprocessinglib.evaluation.metrics

.. autofunction:: generate_dataframes

`JUPYTER NOTEBOOK EXAMPLE <../notebooks/Examples.ipynb>`_

Example
^^^^^^^
Generate the observed and simulated dataframes from the csv file

.. code-block:: python
    :emphasize-lines: 1,3
    :linenos:
    
    from postprocessinglib.evaluation import metrics

    observed_all, simulated_all = metrics.generate_dataframes(csvfpath="MESH_output_streamflow.csv", num_min=365)
    print(observed)
                QOMEAS_05BB001  QOMEAS_05BA001
    YEAR JDAY
    1980 366            10.20            -1.0
    1981 1               9.85            -1.0
         2              10.20            -1.0
         3              10.00            -1.0
         4              10.10            -1.0
    ...                   ...             ...
    2017 361            -1.00            -1.0
         362            -1.00            -1.0
         363            -1.00            -1.0
         364            -1.00            -1.0
         365            -1.00            -1.0
    
    print(simulated)
           QOSIM_05BB001  QOSIM_05BA001
    YEAR JDAY
    1980 366        2.530770       1.006860
    1981 1          2.518999       1.001954
         2          2.507289       0.997078
         3          2.495637       0.992233
         4          2.484073       0.987417
    ...                  ...            ...
    2017 361        4.418050       1.380227
         362        4.393084       1.372171
         363        4.368303       1.364174
         364        4.343699       1.356237
         365        4.319275       1.348359

.. code-block:: python
    :emphasize-lines: 1,3
    :linenos:
    
    from postprocessinglib.evaluation import metrics
   
    observed_from_1992_FEB, simulated_from_1992_FEB = metrics.generate_dataframes(csvfpath="MESH_output_streamflow.csv",
        num_min=365, start_date="1992-02-29")
    
    """ 
    Notice how the index starts from the 29th of February i.e., 31 days in 
    January plus 29 in February equals 60)
    """
    print(observed_from_1992_FEB)  
           QOMEAS_05BB001  QOMEAS_05BA001
    YEAR JDAY
    1992 60              7.37            -1.0
         61              7.36            -1.0
         62              7.56            -1.0
         63              8.01            -1.0
         64              8.21            -1.0
    ...                   ...             ...
    2017 361            -1.00            -1.0
         362            -1.00            -1.0
         363            -1.00            -1.0
         364            -1.00            -1.0
         365            -1.00            -1.0

    print(simulated_from_1992_FEB)
           QOSIM_05BB001  QOSIM_05BA001
    YEAR JDAY
    1992 60         4.960034       1.136079
         61         4.930796       1.129144
         62         4.901721       1.122249
         63         4.872885       1.115409
         64         4.844262       1.108626
    ...                  ...            ...
    2017 361        4.418050       1.380227
         362        4.393084       1.372171
         363        4.368303       1.364174
         364        4.343699       1.356237
         365        4.319275       1.348359

.. code-block:: python
    :emphasize-lines: 1,3
    :linenos:
    
    from postprocessinglib.evaluation import metrics

    observed_till_2000, simulated_till_2000 = metrics.generate_dataframes(csvfpath="MESH_output_streamflow.csv",
        num_min=365, end_date="2000-12-31")
    
    """ 
    Notice how the index ends at December 31st i.e., day 366)
    """
    print(observed_till_2000)
                QOMEAS_05BB001     QOMEAS_05BA001
    YEAR JDAY
    1980 366            10.20            -1.0
    1981 1               9.85            -1.0
         2              10.20            -1.0
         3              10.00            -1.0
         4              10.10            -1.0
    ...                   ...             ...
    2000 362            -1.00            -1.0
         363            -1.00            -1.0
         364            -1.00            -1.0
         365            -1.00            -1.0
         366            -1.00            -1.0

    print(simulated_till_2000
            QOSIM_05BB001  QOSIM_05BA001
    YEAR JDAY
    1980 366        2.530770       1.006860
    1981 1          2.518999       1.001954
         2          2.507289       0.997078
         3          2.495637       0.992233
         4          2.484073       0.987417
    ...                  ...            ...
    2000 362        4.978491       1.455587
         363        4.948421       1.446135
         364        4.918590       1.436760
         365        4.888982       1.427463
         366        4.859608       1.418243


.. code-block:: python
    :emphasize-lines: 1,3
    :linenos:
    
    from postprocessinglib.evaluation import metrics

    observed_January_2000, simulated_January_2000 = metrics.generate_dataframes(csvfpath="MESH_output_streamflow.csv",
        num_min=365, start_date= "2000-1-01", end_date="2000-1-31")

    print(observed_January_2000)
               QOMEAS_05BB001  QOMEAS_05BA001
    YEAR JDAY
    2000 1               -1.0            -1.0
         2               -1.0            -1.0
         3               -1.0            -1.0
         4               -1.0            -1.0
         5               -1.0            -1.0
         6               -1.0            -1.0
         7               -1.0            -1.0
         8               -1.0            -1.0
         9               -1.0            -1.0
         10              -1.0            -1.0
         11              -1.0            -1.0
         12              -1.0            -1.0
         13              -1.0            -1.0
         14              -1.0            -1.0
         15              -1.0            -1.0
         16              -1.0            -1.0
         17              -1.0            -1.0
         18              -1.0            -1.0
         19              -1.0            -1.0
         20              -1.0            -1.0
         21              -1.0            -1.0
         22              -1.0            -1.0
         23              -1.0            -1.0
         24              -1.0            -1.0
         25              -1.0            -1.0
         26              -1.0            -1.0
         27              -1.0            -1.0
         28              -1.0            -1.0
         29              -1.0            -1.0
         30              -1.0            -1.0
         31              -1.0            -1.0

    print(simulated_January_2000)
               QOSIM_05BB001  QOSIM_05BA001
    YEAR JDAY
    2000 1          3.636044       0.986400
         2          3.616069       0.980706
         3          3.596241       0.975054
         4          3.576548       0.969444
         5          3.556993       0.963874
         6          3.537582       0.958345
         7          3.518296       0.952856
         8          3.499155       0.947407
         9          3.480136       0.941998
         10         3.461254       0.936629
         11         3.442506       0.931299
         12         3.423884       0.926007
         13         3.405396       0.920754
         14         3.387026       0.915539
         15         3.368795       0.910362
         16         3.350679       0.905222
         17         3.332695       0.900119
         18         3.314837       0.895053
         19         3.297090       0.890024
         20         3.279476       0.885031
         21         3.261972       0.880074
         22         3.244606       0.875152
         23         3.227338       0.870266
         24         3.210198       0.865415
         25         3.193166       0.860598
         26         3.176266       0.855816
         27         3.159466       0.851068
         28         3.142789       0.846353
         29         3.126214       0.841672
         30         3.109759       0.837024
         31         3.093412       0.832410