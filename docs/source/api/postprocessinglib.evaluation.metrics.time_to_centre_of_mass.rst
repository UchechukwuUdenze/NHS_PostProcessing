time_to_centre_of_mass
======================

.. currentmodule:: postprocessinglib.evaluation.metrics

.. autofunction:: time_to_centre_of_mass

`JUPYTER NOTEBOOK TUTORIAL <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks>`_


Example
^^^^^^^
Calculate the Time to Centre of Mass

.. code-block:: python
    :emphasize-lines: 1,3,36
    :linenos:
    
    from postprocessinglib.evaluation import metrics

    observed, simulated = metrics.generate_dataframes(csvfpath="MESH_output_streamflow.csv", num_min=365)
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


    # Calculating the time to peak
    ttcom = metrics.time_to_centre_of_mass(df=observed, num_stations=1)
    print(ttcom)
        [193.911419943451]