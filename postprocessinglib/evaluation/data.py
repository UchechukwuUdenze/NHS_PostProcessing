"""
The preprocessing module contains all of the functions used alongside the metrics to filter, limit and validate the data 
before it gets evaluated.

"""

import numpy as np
import pandas as pd

from postprocessinglib.utilities import helper as hlp

def station_dataframe(observed: pd.DataFrame, simulated: pd.DataFrame,
               station_num: int) -> pd.DataFrame:
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
    pd.DataFrames:
            The station(s) observed and simulated data

    """

    # validate inputs
    hlp.validate_data(observed, simulated)

    Stations = []
    if station_num <= observed.columns.size:
        for j in range(0, station_num):
            station_df =  observed.copy()
            station_df.drop(station_df.iloc[:, 0:], inplace=True, axis=1)
            station_df = pd.concat([station_df, observed.iloc[:, j], simulated.iloc[:, j]], axis = 1)
            Stations.append(station_df)
        return Stations
    
#### aggregation(weekly, monthly, yearly)(check hydrostats)
# (median, mean, min, max, sum, instantaenous values options)

def seasonal_period(df: pd.DataFrame, daily_period: tuple[str, str],
                              time_range: tuple[str, str]=None) -> pd.DataFrame:
    """Creates a dataframe with a specified seasonal period

    Parameters
    ----------
    merged_dataframe: DataFrame
        A pandas DataFrame with a datetime index and columns containing float type values.
    daily_period: tuple(str, str)
        A list of length two with strings representing the start and end dates of the seasonal period (e.g.
        (01-01, 01-31) for Jan 1 to Jan 31.
    time_range: tuple(str, str)
        A tuple of string values representing the start and end dates of the time range. Format is YYYY-MM-DD.

    Returns
    -------
    pd.Dataframe
        Pandas dataframe that has been truncated to fit the parameters specified for the seasonal period.
    """
    # Making a copy to avoid changing the original df
    df_copy = df.copy()

    if time_range:
        # Setting the time range
        df_copy = df_copy.loc[time_range[0]: time_range[1]]
    
    # Setting a placeholder for the datetime string values
    df_copy.insert(loc=0, column='placeholder', value=df_copy.index.strftime('%m-%d'))

    # getting the start and end of the seasonal period
    start = daily_period[0]
    end = daily_period[1]

    # Getting the seasonal period
    if start < end:
        df_copy = df_copy.loc[(df_copy['placeholder'] >= start) &
                              (df_copy['placeholder'] <= end)]
    else:
        df_copy = df_copy.loc[(df_copy['placeholder'] >= start) |
                              (df_copy['placeholder'] <= end)]
    # Dropping the placeholder
    df_copy = df_copy.drop(columns=['placeholder'])
    
    return df_copy

def daily_aggregate(df: pd.DataFrame, method: str="mean") -> pd.DataFrame:
    """ Returns the daily aggregate value of a given dataframe based
        on the chosen method 

    Parameters
    ---------- 
    df: pd.DataFrame
            A pandas DataFrame with a datetime index and columns containing float type values.
    method: string
            string indicating the method of aggregation
            i.e, mean, min, max, median, sum and instantaenous
            - default is mean

    Returns
    -------
    pd.DataFrame:
            The new dataframe with the values aggregated by day 

    """
    # Check that there is a chosen method else return error
    if not method:
        raise RuntimeError("ERROR: A method of aggregation is required")
    else:
        # Making a copy to avoid changing the original df
        df_copy = df.copy()

        if method == "sum":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m/%d")).sum()
        if method == "mean":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m/%d")).mean()
        if method == "median":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m/%d")).median()
        if method == "min":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m/%d")).min()
        if method == "max":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m/%d")).max()
        if method == "inst":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m/%d")).last() 
    
    return daily_aggr

def weekly_aggregate(df: pd.DataFrame, method: str="mean") -> pd.DataFrame:
    """ Returns the weekly aggregate value of a given dataframe based
        on the chosen method 

    Parameters
    ---------- 
    df: pd.DataFrame
            A pandas DataFrame with a datetime index and columns containing float type values.
    method: string
            string indicating the method of aggregation
            i.e, mean, min, max, median, sum and instantaenous
            - default is mean

    Returns
    -------
    pd.DataFrame:
            The new dataframe with the values aggregated by week 

    """
    # Check that there is a chosen method else return error
    if not method:
        raise RuntimeError("ERROR: A method of aggregation is required")
    else:
        # Making a copy to avoid changing the original df
        df_copy = df.copy()

        if method == "sum":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%W")).sum()
        if method == "mean":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%W")).mean()
        if method == "median":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%W")).median()
        if method == "min":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%W")).min()
        if method == "max":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%W")).max()
        if method == "inst":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%W")).last()    
    
    return weekly_aggr

def monthly_aggregate(df: pd.DataFrame, method: str="mean") -> pd.DataFrame:
    """ Returns the weekly aggregate value of a given dataframe based
        on the chosen method 

    Parameters
    ---------- 
    df: pd.DataFrame
            A pandas DataFrame with a datetime index and columns containing float type values.
    method: string
            string indicating the method of aggregation
            i.e, mean, min, max, median, sum and instantaenous
            - default is mean

    Returns
    -------
    pd.DataFrame:
            The new dataframe with the values aggregated by 
            months of the year 

    """
    # Check that there is a chosen method else return error
    if not method:
        raise RuntimeError("ERROR: A method of aggregation is required")
    else:
        # Making a copy to avoid changing the original df
        df_copy = df.copy()

        if method == "sum":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m")).sum()
        if method == "mean":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m")).mean()
        if method == "median":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m")).median()
        if method == "min":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m")).min()
        if method == "max":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m")).max()
        if method == "inst":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m")).last()    
    
    return monthly_aggr

def yearly_aggregate(df: pd.DataFrame, method: str="mean") -> pd.DataFrame:
    """ Returns the weekly aggregate value of a given dataframe based
        on the chosen method 

    Parameters
    ---------- 
    df: pd.DataFrame
            A pandas DataFrame with a datetime index and columns containing float type values.
    method: string
            string indicating the method of aggregation
            i.e, mean, min, max, median, sum and instantaenous
            - default is mean

    Returns
    -------
    pd.DataFrame:
            The new dataframe with the values aggregated by week 

    """
    # Check that there is a chosen method else return error
    if not method:
        raise RuntimeError("ERROR: A method of aggregation is required")
    else:
        # Making a copy to avoid changing the original df
        df_copy = df.copy()
        
        if method == "sum":
            yearly_aggr = df_copy.groupby(df_copy.index.strftime("%Y")).sum()
        if method == "mean":
            yearly_aggr = df_copy.groupby(df_copy.index.strftime("%Y")).mean()
        if method == "median":
            yearly_aggr = df_copy.groupby(df_copy.index.strftime("%Y")).median()
        if method == "min":
            yearly_aggr = df_copy.groupby(df_copy.index.strftime("%Y")).min()
        if method == "max":
            yearly_aggr = df_copy.groupby(df_copy.index.strftime("%Y")).max()  
        if method == "inst":
            yearly_aggr = df_copy.groupby(df_copy.index.strftime("%Y")).last()  
    
    return yearly_aggr  

def generate_dataframes(csv_fpath: str, warm_up: int = 0, start_date :str = "", end_date: str = "",
                        daily_agg:bool=False, da_method:str="", weekly_agg:bool=False, wa_method:str="",
                        monthly_agg:bool=False, ma_method:str="", yearly_agg:bool=False, ya_method:str="",
                        seasonal_p:bool=False, sp_dperiod:tuple[str, str]=[], sp_time_range:tuple[str, str]=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Function to Generate the required dataframes

    Parameters
    ----------
    csv_fpath : string
            the path to the csv file. It can be relative or absolute
    num_min: int 
            number of days required to "warm up" the system
    start_date: str 
            The date at which you want to start calculating the metric in the
            format yyyy-mm-dd
    end_date: str
            The date at which you want the calculations to end in the
            format yyyy-mm-dd
    daily_agg: bool = False
            If True calculate and return the daily aggregate of the combined dataframes
            using da_method if its available
    da_method: str = ""
            If provided, it determines the method of daily aggregation. It 
            is "mean" by default, see daily_aggregate() function
    weekly_agg: bool = False
            If True calculate and return the weekly aggregate of the combined dataframes
            using wa_method if its available
    wa_method: str = ""
            If provided, it determines the method of weekly aggregation. It 
            is "mean" by default, see weekly_aggregate() function
    monthly_agg: bool = False
            If True calculate and return the monrhly aggregate of the combined dataframes
            using ma_method if its available
    ma_method: str = ""
            If provided, it determines the method of monthly aggregation. It 
            is "mean" by default, see monthly_aggregate() function
    yearly_agg: bool = False
            If True calculate and return the yearly aggregate of the combined dataframes
            using ya_method if its available
    ya_method: str = ""
            If provided, it determines the method of yearly aggregation. It 
            is "mean" by default, see yearly_aggregate() function
    seasonal_p: bool = False
            If True calculate and return a dataframe truncated to fit the parameters specified
            for the seasonal period 
            Requirement:- sp_dperiod.
    sp_dperiod: tuple(str, str)
            A list of length two with strings representing the start and end dates of the seasonal period (e.g.
            (01-01, 01-31) for Jan 1 to Jan 31.
    sp_time_range: tuple(str, str)
            A tuple of string values representing the start and end dates of the time range. Format is YYYY-MM-DD.

    Returns
    -------
    dict{str: pd.dataframe}
            A dictionary containing each Dataframe requested. Its default content is:
            DF = merged dataframe
            DF_SIMULATED = all simulated data
            DF_OBSERVED = all observed data
            
            Depending on which you requested it can also contain:
            DF_DAILY = dataframe aggregated by days of the year
            DF_WEEKLY = dataframe aggregated by the weeks of the year
            DF_MONTHLY = dataframe aggregated by months of the year
            DF_YEARLY = dataframe aggregated by all the years in the data
            DF_CUSTOM = dataframe truncated as per the seasonal period parameters

    """

    if csv_fpath is not None:
        # read the csv into a dataframe making sure to account for unnecessary spaces.
        df = pd.read_csv(csv_fpath, skipinitialspace = True, index_col = ["YEAR", "JDAY"])
    
    # if there are any extra columns at the end of the csv file, remove them
    if len(df.columns) % 2 != 0:
        df.drop(columns=df.columns[-1], inplace = True)

    # Convert the year and jday index to datetime indexing
    start_day = hlp.MultiIndex_to_datetime(df.index[0])
    df.index = pd.to_datetime([i for i in range(len(df.index))], unit='D',origin=pd.Timestamp(start_day))    

    DATAFRAMES = {}
    DATAFRAMES["DF"] = df[warm_up:]
    # Take off the warm up time
    # df = df.replace(-1, np.nan)
    simulated = observed = df[warm_up:].copy()
    simulated.drop(simulated.iloc[:, 0:], inplace=True, axis=1)
    observed.drop(observed.iloc[:, 0:], inplace=True, axis=1)
    for j in range(0, len(df.columns), 2):
        arr1 = df.iloc[warm_up:, j]
        arr2 = df.iloc[warm_up:, j+1]
        observed = pd.concat([observed, arr1], axis = 1)
        simulated = pd.concat([simulated, arr2], axis = 1)

    # splice the dataframes according to the time frame
    if not start_date and end_date:
        # there's an end date but no start date
        simulated = simulated.loc[:end_date]
        observed = observed.loc[:end_date]
    elif not end_date and start_date:
        # there's and end date but no start date
        simulated = simulated.loc[start_date:]
        observed = observed.loc[start_date:]
    elif start_date and end_date:
        # there's a start and end date
        simulated = simulated.loc[start_date:end_date]
        observed = observed.loc[start_date:end_date]
    
    # validate inputs
    hlp.validate_data(observed, simulated)

    DATAFRAMES["DF_SIMULATED"] = simulated
    DATAFRAMES["DF_OBSERVED"] = observed

    # Creating the remaining dataframes based on input
    # 1. Daily aggregate
    if daily_agg and da_method:
        DATAFRAMES["DF_DAILY"] = daily_aggregate(df = DATAFRAMES["DF"], method=da_method)
    elif daily_agg:
        # mean by default
        DATAFRAMES["DF_DAILY"] = daily_aggregate(df = DATAFRAMES["DF"])

    # 2. Weekly aggregate
    if weekly_agg and wa_method:
        DATAFRAMES["DF_WEEKLY"] = weekly_aggregate(df = DATAFRAMES["DF"], method=wa_method)
    elif weekly_agg:
        # mean by default
        DATAFRAMES["DF_WEEKLY"] = weekly_aggregate(df = DATAFRAMES["DF"])

    # 3. Monthly aggregate
    if monthly_agg and ma_method:
        DATAFRAMES["DF_MONTHLY"] = monthly_aggregate(df = DATAFRAMES["DF"], method=ma_method)
    elif monthly_agg:
        # mean by default
        DATAFRAMES["DF_MONTHLY"] = monthly_aggregate(df = DATAFRAMES["DF"])

    # 4.Yearly aggregate
    if yearly_agg and ya_method:
        DATAFRAMES["DF_YEARLY"] = yearly_aggregate(df = DATAFRAMES["DF"], method=ya_method)
    elif yearly_agg:
        # mean by default
        DATAFRAMES["DF_YEARLY"] = yearly_aggregate(df = DATAFRAMES["DF"])

    # 5. Seasonal Period
    if seasonal_p and sp_dperiod == []:
        raise RuntimeError("You cannot calculate a seasonal period without a daily period")
    elif seasonal_p and sp_dperiod and sp_time_range:
        DATAFRAMES["DF_CUSTOM"] = seasonal_period(df = DATAFRAMES["DF"], daily_period=sp_dperiod,
                                                  time_range=sp_time_range)    
    elif seasonal_p and sp_dperiod:
        DATAFRAMES["DF_CUSTOM"] = seasonal_period(df = DATAFRAMES["DF"], daily_period=sp_dperiod)
    
    
    return DATAFRAMES