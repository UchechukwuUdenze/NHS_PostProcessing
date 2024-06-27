"""
The metrics module contains all of the metrics required during the prost processing process.
Each metric has a function and there are heper functions that help with error checking and 
reading and preparatoin of data to be evaluated.  

"""

from collections.abc import Generator
import numpy as np
import pandas as pd

from postprocessinglib.utilities.errors import AllInvalidError


def available_metrics() -> list[int]:
    """ Get a list of currently available metrics

    Returns
    -------
    List[str]
        List of implemented metric names.

    """
    metrics =[
        "MSE", "RMSE", "MAE", "NSE", "KGE", "PBIAS"
    ]
    
    return metrics

def validate_inputs(observed: pd.DataFrame, simulated: pd.DataFrame):
    if not isinstance(observed, pd.DataFrame) or not isinstance(simulated, pd.DataFrame):
        raise ValueError("Both observed and simulated values must be pandas DataFrames.")
    
    if observed.shape != simulated.shape:
        raise RuntimeError("Shapes of observations and simulations must match")

    if (len(observed.shape) < 2) or (observed.shape[1] < 2):
        raise RuntimeError("observed or simulated data is incomplete")


def generate_dfs(csv_fpath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Function to Generate the required dataframes

    Parameters
    ----------
    csv_fpath : string
            the path to the csv file. It can be relative or absolute

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        the observed datframe, the simulated dataframe

    """

    if csv_fpath is not None:
        # read the csv into a dataframe making sure to account for unnecessary spaces.
        df = pd.read_csv(csv_fpath, skipinitialspace = True)
    
    # if there are any extra columns at the end of the csv file, remove them
    if len(df.columns) % 2 != 0:
        df.drop(columns=df.columns[-1], inplace = True) 
        
    simulated = observed = df.iloc[:, 1]
    for j in range(2, len(df.columns), 2):
        arr1 = df.iloc[:, j]
        arr2 = df.iloc[:, j+1]
        observed = pd.concat([observed, arr1], axis = 1)
        simulated = pd.concat([simulated, arr2], axis = 1)

    # validate inputs
    validate_inputs(observed, simulated)
    
    return observed, simulated


def remove_invalid_df(df: pd.DataFrame, station_num: int = 0, station: str = "",
                  neg: int = 0, zero: int = 0, NaN: int = 0,
                  inf: int = 0) -> pd.DataFrame:
    """ Removes the invalid values from a dataframe

    Parameters
    ----------
    df : pd.DataFrame
            the dataframe which you want to remove invalid values from
    station_num : int
            the number referring to the station values we are trying to modify
    neg = 1: int 
            indicates that the negative fields are the inavlid ones
    zero = 1: int 
            indicate that the zero fields are the negative ones
    NaN = 1: int 
            indicate that the empty fields are the invalid ones
    inf = 1: int
            indicate that the inf fields are the invalid ones

    Returns
    -------
    pd.DataFrame: 
            the modified input dataframe 

    """

    if not station and station_num == 0:
        raise ValueError("You must have either a station_num or station variable")
    
    if not station:
        if neg == 1:
            df = df.drop(df[df.iloc[:, station_num] < 0.0].index)
        elif zero == 1:
            df = df.drop(df[df.iloc[:, station_num] == 0.0].index)
        elif NaN == 1:
            df = df.drop(df[df.iloc[:, station_num] == np.nan].index)
        elif inf == 1:
            df = df.drop(df[df.iloc[:, station_num] == np.inf].index)        
        return df

    if neg == 1:
        df = df.drop(df[df[station] < 0.0].index)
    elif zero == 1:
        df = df.drop(df[df[station] == 0.0].index)
    elif NaN == 1:
        df = df.drop(df[df[station] == np.nan].index)
    elif inf == 1:
        df = df.drop(df[df[station] == np.inf].index)        
    return df


def station_df(observed_df: pd.DataFrame, simulated_df: pd.DataFrame,
               station_num: list[int]) -> Generator[pd.DataFrame]:
    """ Extracts a stations data from the observed and simulated 

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Day of the year; 2: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Day of the year; 2: Streamflow Values]
    station_num: List[int]
            array containing the index of the particular stations(s)

    Returns
    -------
    Generator[pd.DataFrames]:
            The stations(s) observed and simulated data

    """

    # validate inputs
    validate_inputs(observed_df, simulated_df)

    if max(station_num) < observed_df.columns.size:
        for j in station_num:
            station_df =  observed_df.iloc[:, 0]
            station_df = pd.concat([station_df, observed_df.iloc[:, j], simulated_df.iloc[:, j]], axis = 1)
            yield station_df.iloc[:]


def mse(observed: pd.DataFrame, simulated: pd.DataFrame, num_stations: int, num_min: int=0) -> float:
    """ Calculates the Mean Square value of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Day of the year; 2: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Day of the year; 2: Streamflow Values]
    num_stations: int
            number of stations in the data
    num_min: int 
            number of days required to "warm up" the system

    Returns
    -------
    float:
        the mean square value of the data

    """     
    # validate inputs
    validate_inputs(observed, simulated)

    if len(observed) <= num_min:
        raise ValueError("Number of days should be greater than the minimum number of days to warm up the system.")

    MSE = []    
    for j in range(1, num_stations+1):            
        summation = np.sum((abs(observed.iloc[num_min:, j] - simulated.iloc[num_min:, j]))**2)
        mse = summation/len(observed)  #dividing summation by total number of values to obtain average    
        MSE.append(mse)
    
    return MSE


def rmse(observed: pd.DataFrame, simulated: pd.DataFrame, num_stations: int,
        num_min: int=0) -> float:
    """ Calculates the Root Mean Square value of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Day of the year; 2: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Day of the year; 2: Streamflow Values]
    num_stations: int
            number of stations in the data
    num_min: int 
            number of days required to "warm up" the system

    Returns
    -------
    float:
        the root mean square value of the data

    """   
    # validate inputs
    validate_inputs(observed, simulated)

    if len(observed) <= num_min:
        raise ValueError("Number of days should be greater than the minimum number of days to warm up the system.")
    
    RMSE =[]
    for j in range(1, num_stations+1):
        summation = np.sum((abs((observed.iloc[num_min:, j]) - simulated.iloc[num_min:, j]))**2)
        rmse = np.sqrt(summation/len(observed)) #dividing summation by total number of values to obtain average    
        RMSE.append(rmse)    

    return RMSE


def mae(observed: pd.DataFrame, simulated: pd.DataFrame, num_stations: int,
        num_min: int=0) -> float:
    """ Calculates the Mean Average value of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Day of the year; 2: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Day of the year; 2: Streamflow Values]
    num_stations: int
            number of stations in the data
    num_min: int 
            number of days required to "warm up" the system

    Returns
    -------
    float:
        the mean average value of the data

    """
    # validate inputs
    validate_inputs(observed, simulated)

    if len(observed) <= num_min:
        raise ValueError("Number of days should be greater than the minimum number of days to warm up the system.")
    
    MAE = []
    for j in range(1, num_stations+1):            
        summation = np.sum(abs(observed.iloc[num_min:, j] - simulated.iloc[num_min:, j]))
        mae = summation/len(observed)  #dividing summation by total number of values to obtain average   
        MAE.append(mae)
    
    return MAE


def nse(observed: pd.DataFrame, simulated: pd.DataFrame, num_stations: int,
        num_min: int=0) -> float:
    """ Calculates the Nash-Sutcliffe Efficiency of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Day of the year; 2: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Day of the year; 2: Streamflow Values]
    num_stations: int
            number of stations in the data
    num_min: int 
            number of days required to "warm up" the system

    Returns
    -------
    float:
        the Nash-Sutcliffe Efficiency of the data

    """       
    # validate inputs
    validate_inputs(observed, simulated)

    if len(observed) <= num_min:
        raise ValueError("Number of days should be greater than the minimum number of days to warm up the system.")

    NSE = []
    for j in range(1, num_stations+1): 
        # Remove the invalid values from that station 
        valid_observed = remove_invalid_df(observed.iloc[num_min:], station_num = j, neg = 1)
        
        num_valid = len(valid_observed.iloc[:, j])
        observed_mean = np.sum(valid_observed.iloc[:, j])
        observed_mean = observed_mean/num_valid

        summation_num = np.sum((abs(valid_observed.iloc[:, j] - simulated.iloc[:, j]))**2)
        summation_denom = np.sum((abs(valid_observed.iloc[:, j] - observed_mean))**2)
        
        nse = (1 - (summation_num/summation_denom))  #dividing summation by total number of values to obtain average
        NSE.append(nse)
        
    return NSE


def kge(observed: pd.DataFrame, simulated: pd.DataFrame, num_stations: int,
        num_min: int=0, scale: list[float]=[1. ,1. ,1.]) -> float:
    """ Calculates the Kling-Gupta Efficiency of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Day of the year; 2: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Day of the year; 2: Streamflow Values]
    num_stations: int
            number of stations in the data
    num_min: int 
            number of days required to "warm up" the system
    scale: list[float, float, float]
            Scale factor for correlation[0], alpha[1], and beta[2] components 
            in the calculation of KGE

    Returns
    -------
    float:
        the Kling-Gupta Efficiency of the data

    """
    # validate inputs
    validate_inputs(observed, simulated)

    if len(observed) <= num_min:
        raise ValueError("Number of days should be greater than the minimum number of days to warm up the system.")

    KGE = []
    for j in range(1, num_stations+1):
        # Remove the invalid values from that station 
        valid_observed = remove_invalid_df(observed.iloc[num_min:], station_num = j, neg = 1)
        
        num_valid = len(valid_observed.iloc[:, j])
        mean_observed = np.sum(valid_observed.iloc[:, j]) 
        mean_simulated = np.sum(simulated.iloc[:, j][valid_observed.iloc[:, j].index])
        mean_observed = mean_observed / num_valid
        mean_simulated = mean_simulated / num_valid
        
        
        std_observed = np.sum((valid_observed.iloc[:, j] - mean_observed)**2) 
        std_simulated = np.sum((simulated.iloc[:, j][valid_observed.iloc[:, j].index] - mean_simulated)**2)
        sum = np.sum((valid_observed.iloc[:, j] - mean_observed) * (simulated.iloc[:, j] - mean_simulated))
        
        # r: Pearson's Correlation Coefficient
        r = sum / np.sqrt(std_simulated * std_observed)
        
        std_observed = np.sqrt(std_observed/(num_valid - 1))
        std_simulated = np.sqrt(std_simulated/(num_valid - 1))

        # a: A term representing the variability of prediction errors,
        # b: A bias term
        b = mean_simulated / mean_observed
        a = std_simulated / std_observed 
        
        # In 2012 the formula was modified so that a(alpha) equals a slightly different value. 
        # Please ensure you are using the right value for your analysis
        # a =  (std_simulated/ mean_simulated)/(std_observed / mean_observed)
        
        kge = 1 - np.sqrt((scale[0]*(r - 1))**2 + (scale[1]*(a - 1))**2 + (scale[2]*(b - 1))**2)
        KGE.append(kge)
    
    return KGE


def bias(observed: pd.DataFrame, simulated: pd.DataFrame, num_stations: int,
        num_min: int=0) -> float:
    """ Calculates the Percentage Bias of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Day of the year; 2: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Day of the year; 2: Streamflow Values]
    num_stations: int
            number of stations in the data
    num_min: int 
            number of days required to "warm up" the system

    Returns
    -------
    float:
        the Percentage Bias of the data

    """    
    # validate inputs
    validate_inputs(observed, simulated)

    if len(observed) <= num_min:
        raise ValueError("Number of days should be greater than the minimum number of days to warm up the system.")
    
    BIAS = []
    for j in range(1, num_stations+1):  
        # Remove the invalid values from that station 
        valid_observed = remove_invalid_df(observed.iloc[num_min:], station_num = j, neg = 1)
        
        bias = np.sum(valid_observed.iloc[:, j] - simulated.iloc[:, j])/np.sum(abs(valid_observed.iloc[:, j]))
        BIAS.append(bias)
    
    return BIAS
        

def calculate_all_metrics(observed: pd.DataFrame, simulated: pd.DataFrame, num_stations: int,
        num_min: int=0) -> dict[str: float]:
    """Calculate all metrics.

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Day of the year; 2: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Day of the year; 2: Streamflow Values]
    num_stations: int
            number of stations in the data
    num_min: int 
            number of days required to "warm up" the system

    Returns
    -------
    dict{str: float}
            A dictionary containing every metric that can be evaluated and
            its result 
            
    """
    # validate inputs
    validate_inputs(observed, simulated)
    parameters = (observed, simulated, num_stations, num_min)

    check_all_invalid(observed, simulated)

    results = {
        "MSE" : mse(*parameters),
        "RMSE" : rmse(*parameters),
        "MAE" : mae(*parameters),
        "NSE" : nse(*parameters),
        "KGE" : kge(*parameters),
        "BIAS" : bias(*parameters),
    }

    return results

def calculate_metrics(observed: pd.DataFrame, simulated: pd.DataFrame, metrices: list[str], num_stations: int,
        num_min: int=0) -> dict[str, float]:
    """Calculate the requested metrics.

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Day of the year; 2: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Day of the year; 2: Streamflow Values]
    metrices: List[str]
            List of metrics to be calculated
    num_stations: int
            number of stations in the data
    num_min: int 
            number of days required to "warm up" the system

    Returns
    -------
    dict{str: float}
            A dictionary containing each metric to be evaluated and its result 
            
    """
    # validate inputs
    validate_inputs(observed, simulated)
    parameters = (observed, simulated, num_stations, num_min)

    if "all" in metrices:
        return calculate_all_metrics(*parameters)
    
    check_all_invalid(observed, simulated)

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
        elif metric.lower() ==  "kge":
            values["KGE"] = kge(*parameters)
        elif metric.lower() ==  "bias":
            values["BIAS"] = bias(*parameters)
        elif metric.lower() == "pbias":
            values["BIAS"] = bias(*parameters)     
        else:
            raise RuntimeError(f"Unknown metric {metric}")
        

    return values


def check_all_invalid(observed: pd.DataFrame, simulated: pd.DataFrame):
    """Check if all observations or simulations are invalid and raise an exception if this is the case.

    Raises
    ------
    AllInvalidError
        If all observations or all simulations are NaN or negative.
    """
    if len(observed.index) == 0:
        raise AllInvalidError("All observed values are NaN")
    if len(simulated.index) == 0:
        raise AllInvalidError("All simulated values are NaN")
    if (observed.values < 0).all():
        raise AllInvalidError("All observed values are invalid(negative)")
    if (simulated.values < 0).all():
        raise AllInvalidError("All simulated values are invalid(negative)")