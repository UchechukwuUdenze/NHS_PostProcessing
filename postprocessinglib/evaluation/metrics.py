"""
The metrics module contains all of the metrics required during the prost processing process.
Each metric has a function and there are hepler functions that help with error checking and 
reading and preparation of data to be evaluated.
This module contains functions evaluate single sample data  such as the 

- The Slope of Flow duration curve ``slope_fdc``,
- Time to peak ``ttp``, 
- Time to centre of mass ``ttcom``, and,
- Spring Pulse Onset Delay ``spod``.

as well as comparison samples between the measured and predicted data such as 

- Mean Square Error ``mse``,
- Percentage Bias ``bias``, and ,
- Kling Gupta Efficiency ``kge``, amongst others.  

Finally we are also able to calculate a number of metrices at the same time or all of them at the same
time using one of the two function below:

- ``calculate_metrics([list of metrics])``
- ``calculate_all_metrics()``

All these functions along side their expected inputs and outputs are shown below:

"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple


from postprocessinglib.utilities import _helper_functions as hlp

def available_metrics() -> list[int]:
    """ Get a list of currently available metrics

    Returns
    -------
    List[str]
        List of implemented metric names.

    Example
    -------
    
    >>> from postprocessinglib.evaluation import metrics
    >>> print(metrics.available_metrics())
        ["MSE - Mean Square Error", "RMSE - Roor Mean Square Error", 
        "MAE - Mean Average Error", "NSE - Nash-Sutcliffe Efficiency ", 
        "NegNSE - Nash-Sutcliffe Efficiency * -1", "LogNSE - Log of Nash-Sutcliffe Efficiency", 
        "NegLogNSE - Log of Nash-Sutcliffe Efficiency * -1",
        "KGE - Kling-Gupta Efficiency", "NegKGE - Kling-Gupta Efficiency * -1", 
        "KGE 2012 - Kling-Gupta Efficiency modified as of 2012", "BIAS- Prcentage Bias", 
        "AbsBIAS - Absolute Value of the Percentage Bias", "TTP - Time to Peak", 
        "TTCoM - Time to Centre of Mass", "SPOD - Spring Pulse ONset Delay", 
        'FDC Slope - Slope of the Flow Duration Curve' ]

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """
    metrics = [
        "MSE - Mean Square Error", "RMSE - Roor Mean Square Error", 
        "MAE - Mean Average Error", "NSE - Nash-Sutcliffe Efficiency ", 
        "NegNSE - Nash-Sutcliffe Efficiency * -1", "LogNSE - Log of Nash-Sutcliffe Efficiency", 
        "NegLogNSE - Log of Nash-Sutcliffe Efficiency * -1",
        "KGE - Kling-Gupta Efficiency", "NegKGE - Kling-Gupta Efficiency * -1", 
        "KGE 2012 - Kling-Gupta Efficiency modified as of 2012", "BIAS- Prcentage Bias", 
        "AbsBIAS - Absolute Value of the Percentage Bias", "TTP - Time to Peak", 
        "TTCoM - Time to Centre of Mass", "SPOD - Spring Pulse ONset Delay", 
        'FDC Slope - Slope of the Flow Duration Curve'
    ]
    
    return metrics

def mse(observed: pd.DataFrame, simulated: Union[pd.DataFrame, List[pd.DataFrame]], stations: list[int]=[]) -> float:
    """ Calculates the Mean Square value of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame or list[pd.DataFrame]
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    pd.DataFrame:
        the mean square value of the data

    Example
    -------
    Calculate the Mean Square Error
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
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

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Mean Square Error
    >>> mse = metrics.mse(observed = obs, simulated = sim)
    >>> print(mse)
                    model1
        Station 1  0.22170
        Station 2  0.08079

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """     
    if (isinstance(simulated, pd.DataFrame)):
        # If simulated is a single DataFrame, convert it to a list of DataFrames
        simulated = [simulated]
    
    # validate inputs
    for i in range(len(simulated)):
        if not isinstance(simulated[i], pd.DataFrame):
            raise ValueError(f"Simulated data at index {i} is not a DataFrame.")
        hlp.validate_data(observed, simulated[i])

    stations_to_process = [s - 1 for s in stations] if stations else list(range(observed.shape[1]))

    mse_results = {}

    for j in stations_to_process:
        valid_observed = hlp.filter_valid_data(observed, station_num=j)

        station_mse = {}
        for k, sim in enumerate(simulated):
            sim_values = sim.loc[valid_observed.index].iloc[:, j]
            obs_values = valid_observed.iloc[:, j]

            mse_val = np.mean((obs_values - sim_values) ** 2)
            station_mse[f"model{k+1}"] = hlp.sig_figs(mse_val, 4)

        # Store per-station result (station index is 1-based)
        mse_results[f"Station {j+1}"] = station_mse

    return pd.DataFrame(mse_results).T  # Transpose so stations are rows


def rmse(observed: pd.DataFrame, simulated: Union[pd.DataFrame, List[pd.DataFrame]], stations: list[int]=[]) -> float:
    """ Calculates the Root Mean Square value of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame or list[pd.DataFrame]
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    pd.DataFrame:
        the root mean square value of the data

    Example
    -------
    Calculate the Root Mean Square Error
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
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

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Root Mean Square Error
    >>> rmse = metrics.rmse(observed = obs, simulated = sim)
    >>> print(rmse)
                   model1
        Station 1  0.3760
        Station 2  0.3398

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """   
    if (isinstance(simulated, pd.DataFrame)):
        # If simulated is a single DataFrame, convert it to a list of DataFrames
        simulated = [simulated]
    
    # validate inputs
    for i in range(len(simulated)):
        if not isinstance(simulated[i], pd.DataFrame):
            raise ValueError(f"Simulated data at index {i} is not a DataFrame.")
        hlp.validate_data(observed, simulated[i])

    stations_to_process = [s - 1 for s in stations] if stations else list(range(observed.shape[1]))

    rmse_results = {}    
    
    for j in stations_to_process:
        valid_observed = hlp.filter_valid_data(observed, station_num=j)

        station_rmse = {}
        for k, sim in enumerate(simulated):
            sim_values = sim.loc[valid_observed.index].iloc[:, j]
            obs_values = valid_observed.iloc[:, j]

            rmse_val = np.sqrt(np.mean((obs_values - sim_values) ** 2))
            station_rmse[f"model{k+1}"] = hlp.sig_figs(rmse_val, 4)

        # Store per-station result (station index is 1-based)
        rmse_results[f"Station {j+1}"] = station_rmse

    return pd.DataFrame(rmse_results).T  # Transpose so stations are rows


def mae(observed: pd.DataFrame, simulated: pd.DataFrame, stations: list[int]=[]) -> float:
    """ Calculates the Mean Average value of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    float:
        the mean average value of the data

    Example
    -------
    Calculate the Mean Average Error
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
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

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Mean Average Error
    >>> mae = metrics.mae(observed = obs, simulated = sim)
    >>> print(mae)
        [0.3132, 0.2709]

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """
    # validate inputs
    hlp.validate_data(observed, simulated)
    
    MAE = []
    
    # If no stations specified, calculate RMSE for all columns
    stations_to_process = stations if stations else range(observed.columns.size)    
    
    for j in stations_to_process:
        # If using 1-indexed stations, adjust by subtracting 1 for 0-indexing
        if stations:
            j = j-1

        # Remove the invalid values from that station 
        valid_observed = hlp.filter_valid_data(observed, station_num = j)
        
        summation = np.sum(abs(valid_observed.iloc[:, j] - simulated.loc[valid_observed.index].iloc[:, j]))
        mae = summation/len(valid_observed)  #dividing summation by total number of values to obtain average   
        MAE.append(hlp.sig_figs(mae, 4))

    return MAE


def nse(observed: pd.DataFrame, simulated: pd.DataFrame, stations: list[int]=[]) -> float:
    """ Calculates the Nash-Sutcliffe Efficiency of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    float:
        the Nash-Sutcliffe Efficiency of the data

    Example
    -------
    Calculate the Nash-Sutcliffe Efficiency
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
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

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Nash-Sutcliffe Efficiency 
    >>> nse = metrics.nse(observed = obs, simulated = sim)
    >>> print(nse)
        [-0.9713, 0.01669]

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """       
    # validate inputs
    hlp.validate_data(observed, simulated)

    NSE = []
    
    # If no stations specified, calculate NSE for all columns
    stations_to_process = stations if stations else range(observed.columns.size)    
    
    for j in stations_to_process:
        # If using 1-indexed stations, adjust by subtracting 1 for 0-indexing
        if stations:
            j = j-1

        # Remove the invalid values from that station 
        valid_observed = hlp.filter_valid_data(observed, station_num = j)
        
        num_valid = len(valid_observed.iloc[:, j])
        observed_mean = np.sum(valid_observed.iloc[:, j])
        observed_mean = observed_mean/num_valid

        summation_num = np.sum((abs(valid_observed.iloc[:, j] - simulated.loc[valid_observed.index].iloc[:, j]))**2)
        summation_denom = np.sum((abs(valid_observed.iloc[:, j] - observed_mean))**2)
        
        nse = (1 - (summation_num/summation_denom))  #dividing summation by total number of values to obtain average
        NSE.append(hlp.sig_figs(nse, 4))
        
    return NSE

def lognse(observed: pd.DataFrame, simulated: pd.DataFrame, stations: list[int]=[]) -> float:
    """ Calculates the Logarithmic Nash-Sutcliffe Efficiency of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    float:
        the Logarithmic Nash-Sutcliffe Efficiency of the data

    Example
    -------
    Calculate the Logarithmic Value of Nash-Sutcliffe Efficiency
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
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

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Log of Nash-Sutcliffe Efficiency 
    >>> lognse = metrics.lognse(observed = obs, simulated = sim)
    >>> print(lognse)
        [-0.4923, -0.4228]

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """       
    # validate inputs
    hlp.validate_data(observed, simulated)

    LOGNSE = []
    
    # If no stations specified, calculate LOGNSE for all columns
    stations_to_process = stations if stations else range(observed.columns.size)    
    
    for j in stations_to_process:
        # If using 1-indexed stations, adjust by subtracting 1 for 0-indexing
        if stations:
            j = j-1

        # Remove the invalid values from that station 
        valid_observed = hlp.filter_valid_data(observed, station_num = j)
        
        num_valid = len(valid_observed.iloc[:, j])
        observed_mean = np.sum(np.log(valid_observed.iloc[:, j]))
        observed_mean = observed_mean/num_valid

        summation_num = np.sum((abs(np.log(valid_observed.iloc[:, j]) - np.log(simulated.loc[valid_observed.index].iloc[:, j])))**2)
        summation_denom = np.sum((abs(np.log(valid_observed.iloc[:, j]) - observed_mean))**2)
        
        lognse = (1 - (summation_num/summation_denom))  #dividing summation by total number of values to obtain average
        LOGNSE.append(hlp.sig_figs(lognse, 4))

    return LOGNSE


def kge(observed: pd.DataFrame, simulated: pd.DataFrame, stations: list[int]=[],
             scale: list[float]=[1. ,1. ,1.]) -> float:
    """ Calculates the Kling-Gupta Efficiency of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data
    scale: list[float, float, float]
            Scale factor for correlation[0], alpha[1], and beta[2] components 
            in the calculation of KGE

    Returns
    -------
    float:
        the Kling-Gupta Efficiency of the data

    Example
    -------
    Calculate the Kling-Gupta Efficiency
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
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

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Kling-Gupta Efficiency 
    >>> kge = metrics.kge(observed = obs, simulated = sim)
    >>> print(kge)
        [-0.02108, 0.4929]
    
    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """
    # validate inputs
    hlp.validate_data(observed, simulated)

    KGE = []
    
    # If no stations specified, calculate KGE for all columns
    stations_to_process = stations if stations else range(observed.columns.size)    
    
    for j in stations_to_process:
        # If using 1-indexed stations, adjust by subtracting 1 for 0-indexing
        if stations:
            j = j-1

        # Remove the invalid values from that station 
        valid_observed = hlp.filter_valid_data(observed, station_num = j)
        
        num_valid = len(valid_observed.iloc[:, j])
        mean_observed = np.sum(valid_observed.iloc[:, j]) 
        mean_simulated = np.sum(simulated.iloc[:, j][valid_observed.iloc[:, j].index])
        mean_observed = mean_observed / num_valid
        mean_simulated = mean_simulated / num_valid
        
        
        std_observed = np.sum((valid_observed.iloc[:, j] - mean_observed)**2) 
        std_simulated = np.sum((simulated.iloc[:, j][valid_observed.iloc[:, j].index] - mean_simulated)**2)
        sum = np.sum((valid_observed.iloc[:, j] - mean_observed) * (simulated.loc[valid_observed.index].iloc[:, j] - mean_simulated))
        
        # r: Pearson's Correlation Coefficient
        r = sum / np.sqrt(std_simulated * std_observed)
        
        std_observed = np.sqrt(std_observed/(num_valid - 1))
        std_simulated = np.sqrt(std_simulated/(num_valid - 1))

        # a: A term representing the variability of prediction errors,
        # b: A bias term
        b = mean_simulated / mean_observed
        a = std_simulated / std_observed
        
        kge = 1 - np.sqrt((scale[0]*(r - 1))**2 + (scale[1]*(a - 1))**2 + (scale[2]*(b - 1))**2)
        KGE.append(hlp.sig_figs(kge, 4))

    return KGE


def kge_2012(observed: pd.DataFrame, simulated: pd.DataFrame, stations: list[int]=[],
             scale: list[float]=[1. ,1. ,1.]) -> float:
    """ Calculates the Kling-Gupta Efficiency of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data
    scale: list[float, float, float]
            Scale factor for correlation[0], alpha[1], and beta[2] components 
            in the calculation of KGE

    Returns
    -------
    float:
        the Kling-Gupta Efficiency of the data

    Note
    ----
    This is different from the regular kge in that this uses the coefficient of Variation
    as its bias term (i.e., std/mean) as oppased to just the mean

    Example
    -------
    Calculate the Kling-Gupta Efficiency
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
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

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Kling-Gupta Efficiency 
    >>> kge_2012 = metrics.kge_2012(observed = obs, simulated = sim)
    >>> print(kge_2012)
        [-0.02567, 0.4894]

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """
    # validate inputs
    hlp.validate_data(observed, simulated)

    KGE = []
    
    # If no stations specified, calculate KGE_2012 for all columns
    stations_to_process = stations if stations else range(observed.columns.size)    
    
    for j in stations_to_process:
        # If using 1-indexed stations, adjust by subtracting 1 for 0-indexing
        if stations:
            j = j-1

        # Remove the invalid values from that station 
        valid_observed = hlp.filter_valid_data(observed, station_num = j)
        
        num_valid = len(valid_observed.iloc[:, j])
        mean_observed = np.sum(valid_observed.iloc[:, j]) 
        mean_simulated = np.sum(simulated.iloc[:, j][valid_observed.iloc[:, j].index])
        mean_observed = mean_observed / num_valid
        mean_simulated = mean_simulated / num_valid
        
        
        std_observed = np.sum((valid_observed.iloc[:, j] - mean_observed)**2) 
        std_simulated = np.sum((simulated.iloc[:, j][valid_observed.iloc[:, j].index] - mean_simulated)**2)
        sum = np.sum((valid_observed.iloc[:, j] - mean_observed) * (simulated.loc[valid_observed.index].iloc[:, j] - mean_simulated))
        
        # r: Pearson's Correlation Coefficient
        r = sum / np.sqrt(std_simulated * std_observed)
        
        std_observed = np.sqrt(std_observed/(num_valid - 1))
        std_simulated = np.sqrt(std_simulated/(num_valid - 1))

        # a: A term representing the variability of prediction errors,
        # b: A bias term
        b = mean_simulated / mean_observed
        a =  (std_simulated/ mean_simulated)/(std_observed / mean_observed)
        
        kge = 1 - np.sqrt((scale[0]*(r - 1))**2 + (scale[1]*(a - 1))**2 + (scale[2]*(b - 1))**2)
        KGE.append(hlp.sig_figs(kge, 4))

    return KGE


def bias(observed: pd.DataFrame, simulated: pd.DataFrame, stations: list[int]=[]) -> float:
    """ Calculates the Percentage Bias of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    float:
        the Percentage Bias of the data

    Example
    -------
    Calculate the Percentage Bias
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
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

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> # Calculate the percentage Bias
    >>> bias = metrics.bias(observed = obs, simulated = sim)
    >>> print(bias)
        [-22.92, 3.873]

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """    
    # validate inputs
    hlp.validate_data(observed, simulated)
    
    BIAS = []
    
    # If no stations specified, calculate BIAS for all columns
    stations_to_process = stations if stations else range(observed.columns.size)    
    
    for j in stations_to_process:
        # If using 1-indexed stations, adjust by subtracting 1 for 0-indexing
        if stations:
            j = j-1

        # Remove the invalid values from that station 
        valid_observed = hlp.filter_valid_data(observed, station_num = j)
        
        bias = np.sum(simulated.iloc[:, j] - valid_observed.iloc[:, j])/np.sum(abs(valid_observed.iloc[:, j])) * 100
        BIAS.append(hlp.sig_figs(bias, 4))

    return BIAS
        

def time_to_peak(df: pd.DataFrame, stations: list[int]=[])->float:
    """ Calculates the time to peak of a given series of data whether observed 
        or simulated

    Parameters
    ----------
    df: pd.DataFrame
            the observed or simulated dataframe
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    int:
        the average time to peak value of the given data

    Example
    -------
    Calculation of the Time to Peak

    >>> from postprocessinglib.evaluation import metrics, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> observed = DATAFRAMES["DF_OBSERVED"] 
    >>> simulated = DATAFRAMES["DF_SIMULATED"]
    >>> print(observed)
                QOMEAS_05BB001  QOMEAS_05BA001
    YEAR JDAY
    1980 366            10.20            NaN
    1981 1               9.85            NaN
         2              10.20            NaN
         3              10.00            NaN
         4              10.10            NaN
    ...                   ...             ...
    2017 361              NaN            NaN
         362              NaN            NaN
         363              NaN            NaN
         364              NaN            NaN
         365              NaN            NaN
    >>> .
    >>> print(simulated)
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

    >>> # Calculating the time to peak
    >>> ttp = metrics.time_to_peak(df=observed)
    >>> print(ttp)
        [167, 171]   

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_ 
    
    """
    TTP = []
    last_year = df.index[-1].year
    if not stations:
        for j in range(0, df.columns.size):
            year = df.index[0].year
            first = 0
            yearly_ttp = []
            while year != last_year:
                # check the number of days
                num_of_days = 365
                if hlp.is_leap_year(year):
                    num_of_days = 366
                
                valid_values = np.sum(np.fromiter((df.index[i].year == year for i in range(first, num_of_days+first)), int))
                
                if valid_values > 200 and np.nansum(df.iloc[first:num_of_days+first, j]) > 0.0:
                    peak_day = np.nanargmax(df.iloc[first:num_of_days+first, j]) + 1
                    yearly_ttp.append(peak_day)
                first += valid_values
                year += 1
            ttp = np.mean(yearly_ttp)
            TTP.append(hlp.sig_figs(ttp, 3))
    else:
        for j in stations:
            # Adjust for zero indexing
            j -= 1

            year = df.index[0].year
            first = 0
            yearly_ttp = []
            while year != last_year:
                # check the number of days
                num_of_days = 365
                if hlp.is_leap_year(year):
                    num_of_days = 366
                
                valid_values = np.sum(np.fromiter((df.index[i].year == year for i in range(first, num_of_days+first)), int))
                
                if valid_values > 200 and np.nansum(df.iloc[first:num_of_days+first, j]) > 0.0:
                    peak_day = np.nanargmax(df.iloc[first:num_of_days+first, j]) + 1
                    yearly_ttp.append(peak_day)
                first += valid_values
                year += 1
            ttp = np.mean(yearly_ttp)
            TTP.append(hlp.sig_figs(ttp, 3))

    return TTP

def time_to_centre_of_mass(df: pd.DataFrame, stations: list[int]=[])->float:
    """ Calculates the time it takes to obtain 50% of the stream flow in a given year

    Parameters
    ----------
    df: pd.DataFrame
            the observed or simulated dataframe
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    int:
        the average time to the centre of mass for the station

    Example
    -------
    Calculation of the time to center of mass

    >>> from postprocessinglib.evaluation import metrics, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> observed = DATAFRAMES["DF_OBSERVED"] 
    >>> simulated = DATAFRAMES["DF_SIMULATED"]
    >>> print(observed)
                QOMEAS_05BB001  QOMEAS_05BA001
    YEAR JDAY
    1980 366            10.20            NaN
    1981 1               9.85            NaN
         2              10.20            NaN
         3              10.00            NaN
         4              10.10            NaN
    ...                   ...             ...
    2017 361              NaN            NaN
         362              NaN            NaN
         363              NaN            NaN
         364              NaN            NaN
         365              NaN            NaN
    >>> .
    >>> print(simulated)
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

    >>> # Calculating the time to center of mass
    >>> ttcom = metrics.time_to_centre_of_mass(df=observed)
    >>> print(ttcom)
        [194, 203]

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """
    TTCoM = []
    last_year = df.index[-1].year
    if not stations:
        for j in range(0, df.columns.size):
            year = df.index[0].year
            first = 0
            yearly_ttcom = []
            while year != last_year:
                # check the number of days
                num_of_days = 365
                if hlp.is_leap_year(year):
                    num_of_days = 366

                valid_values = np.sum(np.fromiter((df.index[i].year == year for i in range(first, num_of_days+first)), int))
                
                if valid_values > 200 and np.nansum(df.iloc[first:num_of_days+first, j]) > 0.0:
                    CoM = np.sum(np.arange(1, num_of_days+1) * df.iloc[first:num_of_days+first, j])
                    CoM = CoM / np.nansum(df.iloc[first:num_of_days+first, j])
                    yearly_ttcom.append(CoM)
                first += valid_values
                year += 1
            ttcom = np.mean(yearly_ttcom)
            TTCoM.append(hlp.sig_figs(ttcom, 3))
    else:
        for j in stations:
            # Adjust for zero indexing
            j -= 1

            year = df.index[0].year
            first = 0
            yearly_ttcom = []
            while year != last_year:
                # check the number of days
                num_of_days = 365
                if hlp.is_leap_year(year):
                    num_of_days = 366

                valid_values = np.sum(np.fromiter((df.index[i].year == year for i in range(first, num_of_days+first)), int))
                
                if valid_values > 200 and np.nansum(df.iloc[first:num_of_days+first, j]) > 0.0:
                    CoM = np.sum(np.arange(1, num_of_days+1) * df.iloc[first:num_of_days+first, j])
                    CoM = CoM / np.nansum(df.iloc[first:num_of_days+first, j])
                    yearly_ttcom.append(CoM)
                first += valid_values
                year += 1
            ttcom = np.mean(yearly_ttcom)
            TTCoM.append(hlp.sig_figs(ttcom, 3))

    return TTCoM

def SpringPulseOnset(df: pd.DataFrame, stations: list[int]=[])->int:
    """ Calculates when spring start i.e., the beginning of snowmelt

    Parameters
    ----------
    df: pd.DataFrame
            the observed or simulated dataframe
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    int:
        the average time it takes till when snowmelt begins 

    Example
    -------
    Calculation of the SpringPulseOnset

    >>> from postprocessinglib.evaluation import metrics, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> observed = DATAFRAMES["DF_OBSERVED"] 
    >>> simulated = DATAFRAMES["DF_SIMULATED"]
    >>> print(observed)
                QOMEAS_05BB001  QOMEAS_05BA001
    YEAR JDAY
    1980 366            10.20            NaN
    1981 1               9.85            NaN
         2              10.20            NaN
         3              10.00            NaN
         4              10.10            NaN
    ...                   ...             ...
    2017 361              NaN            NaN
         362              NaN            NaN
         363              NaN            NaN
         364              NaN            NaN
         365              NaN            NaN
    >>> .
    >>> print(simulated)
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

    >>> # Calculating the spring pulse onset day
    >>> spod = metrics.SpringPulseOnset(df=simulated)
    >>> print(spod)
        [136, 143]

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """
    SPOD = []
    last_year = df.index[-1].year
    if not stations:
        for j in range(0, df.columns.size):
            year = df.index[0].year
            first = 0
            yearly_spod = []
            while year != last_year:
                # check the number of days
                num_of_days = 365
                if hlp.is_leap_year(year):
                    num_of_days = 366

                # Check for number of days in the year
                valid_values = np.sum(np.fromiter((df.index[i].year == year for i in range(first, num_of_days+first)), int))
                # print(f"valid values {valid_values}")

                if valid_values > 200 and np.nansum(df.iloc[first:first+valid_values, j]) > 0.0 and not pd.isna(df.iloc[first:first+valid_values, j]).any():
                    # print(np.sum(df.iloc[first:first+valid_values, j]))
                    mean = np.nanmean(df.iloc[first:valid_values+first, j])
                    # print(mean)
                    minimum_cumulative = 1.0E38         # Some Arbitrarily large number
                    cumulative = 0
                    onset_day = 0
                    for index in range(first, valid_values+first):
                        # if not np.isnan(df.iloc[index, j]):
                        cumulative += (df.iloc[index, j] - mean)
                        if cumulative < minimum_cumulative:
                            minimum_cumulative = cumulative
                            # onset_day = (index % valid_values) + (num_of_days-valid_values) + 1
                            onset_day = (index % num_of_days) + 1
                    yearly_spod.append(onset_day)
                    # print(yearly_spod)
                    # print("/n")
                first += valid_values
                year += 1          
            spod = np.mean(yearly_spod)
            SPOD.append(hlp.sig_figs(spod, 3))
            SPOD.append(spod)
    else:
        for j in stations:
            # Adjust for zero indexing
            j -= 1

            year = df.index[0].year
            first = 0
            yearly_spod = []
            while year != last_year:
                # check the number of days
                num_of_days = 365
                if hlp.is_leap_year(year):
                    num_of_days = 366

                # Check for number of days in the year
                valid_values = np.sum(np.fromiter((df.index[i].year == year for i in range(first, num_of_days+first)), int))
                # print(f"valid values {valid_values}")

                if valid_values > 200 and np.nansum(df.iloc[first:first+valid_values, j]) > 0.0 and not pd.isna(df.iloc[first:first+valid_values, j]).any():
                    # print(np.sum(df.iloc[first:first+valid_values, j]))
                    mean = np.nanmean(df.iloc[first:valid_values+first, j])
                    # print(mean)
                    minimum_cumulative = 1.0E38         # Some Arbitrarily large number
                    cumulative = 0
                    onset_day = 0
                    for index in range(first, valid_values+first):
                        # if not np.isnan(df.iloc[index, j]):
                        cumulative += (df.iloc[index, j] - mean)
                        if cumulative < minimum_cumulative:
                            minimum_cumulative = cumulative
                            # onset_day = (index % valid_values) + (num_of_days-valid_values) + 1
                            onset_day = (index % num_of_days) + 1
                    yearly_spod.append(onset_day)
                    # print(yearly_spod)
                    # print("/n")
                first += valid_values
                year += 1          
            spod = np.mean(yearly_spod)
            SPOD.append(hlp.sig_figs(spod, 3))

    return SPOD

def slope_fdc(df: pd.DataFrame, percentiles: tuple[float, float] = (33, 66), stations: list[int] = []) -> list[float]:
    """
    Calculates the slope of the Flow Duration Curve (FDC).

    Parameters
    ----------
    df: pd.DataFrame
        Streamflow values for calculating the FDC slope. Each column corresponds to a station.
    percentiles: tuple[float, float]
        Percentiles used for slope calculation (e.g., (33, 66) for 33rd and 66th percentiles).
    stations: list[int]
        List of station indices (1-indexed) for which to calculate the slope. If empty, all stations are processed.

    Returns
    -------
    list[float]
        Slope of the FDC for each station.

    Example
    -------
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> from metrics import slope_fdc
    >>> data = pd.DataFrame({
    >>>     "Station1": [1.2, 0.8, 0.6, 0.4, 0.2],
    >>>     "Station2": [2.0, 1.5, 1.0, 0.5, 0.2]
    >>> })
    >>> slope_fdc(df=data, percentiles=(33, 66))
    [0.693, 0.847]
    """
    slopes = []
    stations_to_process = stations if stations else range(df.shape[1])

    for j in stations_to_process:
        # Adjust for 0-indexing if stations are provided
        col_index = j - 1 if stations else j

        # Calculate the required percentiles
        q33 = df.iloc[:, col_index].quantile(percentiles[0] / 100)
        q66 = df.iloc[:, col_index].quantile(percentiles[1] / 100)

        # Compute the slope
        slope = (np.log(q66) - np.log(q33)) / (percentiles[1] / 100 - percentiles[0] / 100)
        slopes.append(round(slope, 4))  # Round to 4 significant figures

    return slopes


def calculate_all_metrics(observed: pd.DataFrame, simulated: pd.DataFrame, stations: list[int]=[],
                          format: str="", out: str='metrics_out') -> dict[str, float]:
    """Calculate all metrics.

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data
    format: str
            used to indicate that you want the output to be saved to a output file who's
            name is specified by the 'out' parameter
    out: str
            used in tandem with the 'format' parameter to specify the name of the output file.
            it is 'metrics_out.{format}' by default


    Returns
    -------
    dict[str, float]
            A dictionary containing every metric that can be evaluated and
            its result

    Example
    -------
    Calculation of all available metrics

    >>> from postprocessinglib.evaluation import metrics, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> print(metrics.calculate_all_metrics(observed=DATAFRAMES["DF_OBSERVED"], simulated=DATAFRAMES["DF_SIMULATED"]))
        {'MSE': [1890], 'RMSE': [43.47], 'MAE': [25.14], 'NSE': [0.09948],
        'NegNSE': [-0.09948], 'LogNSE': [-0.3342], 'NegLogNSE': [0.3342], 'KGE': [0.4392],
        'NegKGE': [-0.4392], 'KGE 2012': [0.3130], 'BIAS': [-34.60], 'AbsBIAS': [34.60],
        'TTP_obs': [155], 'TTP_sim': [181], 'TTCoM_obs': [185], 'TTCoM_sim': [191], 'SPOD_obs': [72.0],
        'SPOD_sim': [76.1]} 

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_
            
    """
    # validate inputs
    hlp.validate_data(observed, simulated)
    parameters = (observed, simulated, stations)

    hlp.check_valid_dataframe(observed, simulated)

    results = {
        "MSE" : mse(*parameters),
        "RMSE" : rmse(*parameters),
        "MAE" : mae(*parameters),
        "NSE" : nse(*parameters),
        "NegNSE" : [-x for x in nse(*parameters)],
        "LogNSE" : lognse(*parameters),
        "NegLogNSE" : [-x for x in lognse(*parameters)],
        "KGE" : kge(*parameters),
        "NegKGE" : [-x for x in kge(*parameters)],
        "KGE 2012" : kge_2012(*parameters),
        "BIAS" : bias(*parameters),
        "AbsBIAS" : list(map(abs, bias(*parameters))), 
        "TTP_obs" : time_to_peak(observed,stations),
        "TTP_sim" : time_to_peak(simulated, stations),
        "TTCoM_obs" : time_to_centre_of_mass(observed, stations),
        "TTCoM_sim" : time_to_centre_of_mass(simulated, stations),
        "SPOD_obs" : SpringPulseOnset(observed, stations),
        "SPOD_sim" : SpringPulseOnset(simulated, stations),
        "FDC_Slope_obs": slope_fdc(observed, stations=stations),
        "FDC_Slope_sim": slope_fdc(simulated, stations=stations),
    }

    # Check for a specified format, else print to screen
    if format:
        val = pd.DataFrame(results)
        val.index = val.index+1 # so the index starts form 1 and not 0
        if format == "txt":
            val_txt = val.to_csv(sep='\t', index=False, lineterminator='\n')
            lines = val_txt.split('\n')
            columns = [line.split('\t') for line in lines if line]
            formatted_lines = []
            for line in columns:
                formatted_line = ''.join(f'{col:<{12}}' for col in line)
                formatted_lines.append(formatted_line)            
            # Join the formatted lines into a single string
            formatted_str = '\n'.join(formatted_lines)
            with open(out+"."+format, "w") as file:
                file.write(formatted_str)
            file.close()
            print(f"See {out}.{format} file in directory")
        elif format == "csv":
            val.to_csv(f"{out}.{format}")
            print(f"See {out}.{format} file in directory")            
        else:
            print("unknown or uncoded format - " + format)
    else:
        return results

def calculate_metrics(observed: pd.DataFrame, simulated: pd.DataFrame, metrices: list[str],
                      stations: list[int]=[], format: str="", out: str='metrics_out') -> dict[str, float]:
    """Calculate the requested metrics.

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data
    metrices: List[str]
            List of metrics to be calculated
    format: str
            used to indicate that you want the output to be saved to a output file who's
            name is specified by the 'out' parameter
    out: str
            used in tandem with the 'format' parameter to specify the name of the output file.
            it is 'metrics_out.{format}' by default

    Returns
    -------
    dict[str, float]
            A dictionary containing each metric to be evaluated and its result 

    Example
    -------
    Calculation of a list of metrics

    >>> from postprocessinglib.evaluation import metrics, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> list_of_metrices = ["MSE", "NSE", "KGE 2012"]
    >>> print(metrics.calculate_metrics(observed=DATAFRAMES["DF_OBSERVED"], simulated=DATAFRAMES["DF_SIMULATED"], metrices=list_of_metrics))
        {'MSE': [1890, 665.9], 'NSE': [0.09948, -3.583], 'KGE 2012': [0.3130, -0.1483]} 

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_
               
    """
    # validate inputs
    hlp.validate_data(observed, simulated)
    parameters = (observed, simulated, stations)

    if "all" in metrices:
        return calculate_all_metrics(*parameters)
    
    hlp.check_valid_dataframe(observed, simulated)

    values = {}
    for metric in metrices:
        if metric.lower() ==  "mse":
            values["MSE"] = mse(*parameters)
        elif metric.lower() ==  "rmse":
            values["RMSE"] = rmse(*parameters)
        elif metric.lower() ==  "mae":
            values["MAE"] = mae(*parameters)
        elif metric.lower() ==  "nse":
            values["NSE"] = nse(*parameters)
        elif metric.lower() ==  "negnse":
            values["NegNSE"] = [-x for x in nse(*parameters)]
        elif metric.lower() ==  "lognse":
            values["LogNSE"] = lognse(*parameters)
        elif metric.lower() ==  "neglognse":
            values["NegLogNSE"] = [-x for x in lognse(*parameters)]
        elif metric.lower() ==  "kge":
            values["KGE"] = kge(*parameters)
        elif metric.lower() ==  "negkge":
            values["NegKGE"] = [-x for x in kge(*parameters)]
        elif metric.lower() ==  "kge 2012":
            values["KGE 2012"] = kge_2012(*parameters)
        elif metric.lower() ==  "bias":
            values["BIAS"] = bias(*parameters)
        elif metric.lower() == "pbias":
            values["BIAS"] = bias(*parameters)
        elif metric.lower() ==  "absbias":
            values["AbsBIAS"] = list(map(abs, bias(*parameters))),
        elif metric.lower() == "ttp_obs":
            values["TTP_obs"] = time_to_peak(observed, stations)
        elif metric.lower() == "ttp_sim":
            values["TTP_sim"] = time_to_peak(simulated, stations)
        elif metric.lower() == "ttcom_obs":
            values["TTCoM_obs"] = time_to_centre_of_mass(observed, stations)
        elif metric.lower() == "ttcom_sim":
            values["TTCoM_sim"] = time_to_centre_of_mass(simulated, stations)
        elif metric.lower() == "spod_obs":
            values["SPOD_obs"] = SpringPulseOnset(observed, stations)
        elif metric.lower() == "spod_sim":
            values["SPOD_sim"] = SpringPulseOnset(simulated, stations)
        elif metric.lower() == "fdc_obs":
            values["FDC_Slope_obs"] = slope_fdc(observed, stations=stations)
        elif metric.lower() == "fdc_sim":
            values["FDC_Slope_sim"] = slope_fdc(simulated, stations=stations)
        else:
            raise RuntimeError(f"Unknown metric {metric}")
        
    # Check for a specified format, else print to screen
    if format:
        val = pd.DataFrame(values)
        val.index = val.index+1 # so the index starts form 1 and not 0
        if format == "txt":
            val_txt = val.to_csv(sep='\t', index=False, lineterminator='\n')
            lines = val_txt.split('\n')
            columns = [line.split('\t') for line in lines if line]
            formatted_lines = []
            for line in columns:
                formatted_line = ''.join(f'{col:<{12}}' for col in line)
                formatted_lines.append(formatted_line)            
            # Join the formatted lines into a single string
            formatted_str = '\n'.join(formatted_lines)
            with open(out+"."+format, "w") as file:
                file.write(formatted_str)
            file.close()
            print(f"See {out}.{format} file in directory")
        elif format == "csv":
            val.to_csv(f"{out}.{format}")
            print(f"See {out}.{format} file in directory")            
        else:
            print("unknown or uncoded format - " + format)
    else:
        return values
