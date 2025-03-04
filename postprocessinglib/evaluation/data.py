"""
The preprocessing module contains all of the functions used alongside the metrics to filter, limit and validate the data 
before it gets evaluated.

"""

import numpy as np
import pandas as pd
import re

from postprocessinglib.utilities import _helper_functions as hlp

def station_dataframe(observed: pd.DataFrame, simulated: pd.DataFrame,
               stations: list[int]=[]) -> list[pd.DataFrame]:
    """ Extracts each station's data from the observed and simulated 

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
    list[pd.DataFrame]:
        Each station's observed and simulated data in a single dataframe -
        in a list

    Example
    -------
    Extraction of the Data from Individual Stations

    >>> from postprocessinglib.evaluation import data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> observed = DATAFRAMES["DF_OBSERVED"] 
    >>> simulated = DATAFRAMES["DF_SIMULATED"]
    >>> STATIONS = data.station_dataframe(observed=observed, simulated=simulated)
    >>> for station in STATIONS:
    >>>     print(station)
                    QOMEAS_05BB001  QOSIM_05BB001
        1980-12-31           10.20       2.530770
        1981-01-01            9.85       2.518999
        1981-01-02           10.20       2.507289
        1981-01-03           10.00       2.495637
        1981-01-04           10.10       2.484073
        ...                    ...            ...
        2017-12-27             NaN       4.418050
        2017-12-28             NaN       4.393084
        2017-12-29             NaN       4.368303
        2017-12-30             NaN       4.343699
        2017-12-31             NaN       4.319275
        [13515 rows x 2 columns]
                    QOMEAS_05BA001  QOSIM_05BA001
        1980-12-31             NaN       1.006860
        1981-01-01             NaN       1.001954
        1981-01-02             NaN       0.997078
        1981-01-03             NaN       0.992233
        1981-01-04             NaN       0.987417
        ...                    ...            ...
        2017-12-27             NaN       1.380227
        2017-12-28             NaN       1.372171
        2017-12-29             NaN       1.364174
        2017-12-30             NaN       1.356237
        2017-12-31             NaN       1.348359
        [13515 rows x 2 columns]

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-data-manipulation.ipynb>`_

    """

    # validate inputs
    hlp.validate_data(observed, simulated)

    Stations = []

    # If no stations specified, Seperate all columns
    stations_to_process = stations if stations else range(observed.columns.size)

    for j in stations_to_process:
        # If using 1-indexed stations, adjust by subtracting 1 for 0-indexing
        if stations:
            j = j-1

            station_df =  observed.copy()
            station_df.drop(station_df.iloc[:, 0:], inplace=True, axis=1)
            station_df = pd.concat([station_df, observed.iloc[:, j], simulated.iloc[:, j]], axis = 1)
            Stations.append(station_df)
        return Stations


def long_term_seasonal(df: pd.DataFrame, method: str= 'mean')-> pd.DataFrame:
    """ 
    Computes the long-term seasonal aggregate values of a given DataFrame by 
    applying the specified aggregation method to each day across all years in the 
    provided time period. The resulting data is aggregated into a single year (1 to 366 days).

    Parameters
    ---------- 
    df: pd.DataFrame
            A pandas DataFrame with a datetime index and columns containing float type values.
            Each column represents a time series to be aggregated.
    method: str, optional
        The aggregation method to apply across all years for each specific day of the year.
        Supported methods include:

        - 'mean': Calculate the mean value of that specific day (e.g., January 1st) 
          across all years in the dataset (default).
        - 'min': Calculate the minimum value of that specific day across all years.
        - 'max': Calculate the maximum value of that specific day across all years.
        - 'median': Calculate the median value of that specific day across all years.
        - 'sum': Calculate the sum of that specific day across all years.
        - 'QX': Calculate a specific quantile, where X is a number between 0 and 100
          (e.g., 'Q75' for the 75th percentile). Use uppercase 'Q' for quantiles.
        
        Default is mean

    Returns
    -------
    pd.DataFrame:
        A DataFrame with 366 rows (representing days of the year, including February 29th) 
        and the same columns as the input. Each row represents the aggregated value for 
        that specific day across all years.

    Examples
    --------
    Extraction of a Long term seasonal aggregation

    >>> from postprocessinglib.evaluation import data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> merged_df = DATAFRAMES["DF"]
    >>> print(merged_df)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980-12-31           10.20       2.530770             NaN       1.006860
    1981-01-01            9.85       2.518999             NaN       1.001954
    1981-01-02           10.20       2.507289             NaN       0.997078
    1981-01-03           10.00       2.495637             NaN       0.992233
    1981-01-04           10.10       2.484073             NaN       0.987417
    ...                    ...            ...             ...            ...
    2017-12-27             NaN       4.418050             NaN       1.380227
    2017-12-28             NaN       4.393084             NaN       1.372171
    2017-12-29             NaN       4.368303             NaN       1.364174
    2017-12-30             NaN       4.343699             NaN       1.356237
    2017-12-31             NaN       4.319275             NaN       1.348359

    >>> # Extract the long term Seasonal Aggregation
    >>> long_term_seasonal = data.long_term_seasonal(df=merged_df) # Recall the default is mean
    >>> print(long_term_seasonal)
          QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    jday
    1           9.446471       4.037666             NaN       1.130686
    2           9.428125       4.014474             NaN       1.123915
    3           9.660625       3.991451             NaN       1.117196
    4           9.804375       3.968602             NaN       1.110529
    5           9.787500       3.945921             NaN       1.103913
    ...              ...            ...             ...            ...
    362         9.942500       4.188140             NaN       1.169614
    363         9.695000       4.163847             NaN       1.162533
    364         9.633125       4.139735             NaN       1.155507
    365         9.516875       4.115805             NaN       1.148535
    366         9.936000       4.243073             NaN       1.173174

    >>> # Obtain the Upper Quartile - Q75
    >>> long_term_seasonal = data.long_term_seasonal(df=merged_df, method = 'Q75')
    >>> print(long_term_seasonal)
          QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    jday
    1             10.100       4.830453             NaN       1.315370
    2             10.200       4.801530             NaN       1.306986
    3             10.550       4.772831             NaN       1.298670
    4             10.500       4.744344             NaN       1.290422
    5             10.700       4.716085             NaN       1.282241
    ...              ...            ...             ...            ...
    362           11.175       4.978491             NaN       1.372171
    363           11.075       4.948421             NaN       1.364174
    364           11.100       4.918590             NaN       1.356237
    365           10.775       4.888982             NaN       1.348359
    366           11.300       4.765379             NaN       1.317393
    
    """
    # Check that there is a chosen method else return error
    if not method:
        raise RuntimeError("ERROR: A method of aggregation is required")
    else:
        # Making a copy to avoid changing the original df
        df_copy = df.copy()
    
        df_copy.index = df_copy.index.strftime("%Y-%m-%d")
        df_copy.index = pd.MultiIndex.from_tuples([hlp.datetime_to_index(index) for index in df_copy.index],
                                                names=('year', 'jday'))

        # Making a pattern to check with
        pattern = r'^[qQ]\d{2}(\.\d{1,2})?$' 
        
        if method == 'mean': 
            df_copy = df_copy.groupby(level = 'jday').mean()
        elif method == 'min':
            df_copy = df_copy.groupby(level = 'jday').min()
        elif method == 'max':
            df_copy = df_copy.groupby(level = 'jday').max()
        elif method == 'median':
            df_copy = df_copy.groupby(level = 'jday').median()
        elif method == 'sum':
            df_copy = df_copy.groupby(level = 'jday').sum()
        # Check the quartiles with the matching pattern 
        elif re.match(pattern=pattern, string=method):
                df_copy = df_copy.groupby(level = 'jday').quantile(float(re.search(r'\d+', method).group())/100)
    
    return df_copy

def seasonal_period(df: pd.DataFrame, daily_period: tuple[str, str],
                    subset: tuple[str, str]=None, years: list[int]=None) -> pd.DataFrame:
    """Creates a dataframe with a specified seasonal period

    Parameters
    ----------
    merged_dataframe: DataFrame
        A pandas DataFrame with a datetime index and columns containing float type values.
    daily_period: tuple(str, str)
        A tuple of two strings representing the start and end dates of the seasonal period (e.g.
        (01-01, 01-31) for Jan 1 to Jan 31.
    subset: tuple(str, str)
        A tuple of string values representing the start and end dates of the subset. Format is YYYY-MM-DD.
    years: list[int]
        A list of years to filter the dataframe by. If provided, only data from these years will be included.

    Returns
    -------
    pd.Dataframe
        Pandas dataframe that has been truncated to fit the parameters specified for the seasonal period.
    
    Examples
    --------
    Extraction of a Seasonal period

    >>> from postprocessinglib.evaluation import data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> merged_df = DATAFRAMES["DF"]
    >>> print(merged_df)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980-12-31           10.20       2.530770             NaN       1.006860
    1981-01-01            9.85       2.518999             NaN       1.001954
    1981-01-02           10.20       2.507289             NaN       0.997078
    1981-01-03           10.00       2.495637             NaN       0.992233
    1981-01-04           10.10       2.484073             NaN       0.987417
    ...                    ...            ...             ...            ...
    2017-12-27             NaN       4.418050             NaN       1.380227
    2017-12-28             NaN       4.393084             NaN       1.372171
    2017-12-29             NaN       4.368303             NaN       1.364174
    2017-12-30             NaN       4.343699             NaN       1.356237
    2017-12-31             NaN       4.319275             NaN       1.348359
    
    >>> # Extract the time period - January 1st till 31st - using the subset
    >>> seasonal_p = data.seasonal_period(df=merged_df, daily_period=('01-01', '01-31'),
                            subset = ('1981-01-01', '1985-12-31'))
    >>> print(seasonal_p)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1981-01-01            9.85       2.518999             NaN       1.001954
    1981-01-02           10.20       2.507289             NaN       0.997078
    1981-01-03           10.00       2.495637             NaN       0.992233
    1981-01-04           10.10       2.484073             NaN       0.987417
    1981-01-05            9.99       2.472571             NaN       0.982631
    ...                    ...            ...              ...            ...
    1985-01-27           11.40       2.734883             NaN       0.809116
    1985-01-28           11.60       2.721414             NaN       0.805189
    1985-01-29           11.70       2.708047             NaN       0.801287
    1985-01-30           11.60       2.694749             NaN       0.797410
    1985-01-31           11.60       2.681550             NaN       0.793556

    >>> # Extract the time period - January 1st till 10th - using the years
    >>> seasonal_p = data.seasonal_period(df=merged_df, daily_period=('01-01', '01-10'),
                            year = [1981, 1983, 1985])
    >>> print(seasonal_p)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1981-01-01            9.85       2.518999             NaN       1.001954
    1981-01-02           10.20       2.507289             NaN       0.997078
    1981-01-03           10.00       2.495637             NaN       0.992233
    1981-01-04           10.10       2.484073             NaN       0.987417
    1981-01-05            9.99       2.472571             NaN       0.982631
    1981-01-06            9.69       2.461128             NaN       0.977875
    1981-01-07            9.51       2.449758             NaN       0.973148
    1981-01-08            8.90       2.438459             NaN       0.968450
    1981-01-09            8.70       2.427217             NaN       0.963778
    1981-01-10            9.00       2.416050             NaN       0.959137
    1983-01-01            8.98       5.371416             NaN       2.441398
    1983-01-02            8.89       5.340234             NaN       2.426411
    1983-01-03            9.12       5.309281             NaN       2.411540
    1983-01-04            9.37       5.278562             NaN       2.396784
    1983-01-05            9.40       5.248067             NaN       2.382142
    1983-01-06            9.54       5.217788             NaN       2.367613
    1983-01-07            9.44       5.187746             NaN       2.353197
    1983-01-08            9.21       5.157917             NaN       2.338891
    1983-01-09            9.03       5.128305             NaN       2.324696
    1983-01-10            8.35       5.098919             NaN       2.310610
    1985-01-01           10.10       3.116840             NaN       0.920429
    1985-01-02           10.20       3.100937             NaN       0.915796
    1985-01-03           10.70       3.085116             NaN       0.911193
    1985-01-04            9.90       3.069416             NaN       0.906620
    1985-01-05            9.51       3.053805             NaN       0.902076
    1985-01-06            9.27       3.038310             NaN       0.897561
    1985-01-07            9.55       3.022904             NaN       0.893076
    1985-01-08           10.10       3.007609             NaN       0.888619
    1985-01-09           10.00       2.992402             NaN       0.884191
    1985-01-10           10.00       2.977300             NaN       0.879791

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-data-manipulation.ipynb>`_
    
    """
    # Making a copy to avoid changing the original df
    df_copy = df.copy()

    if subset:
        # Setting the time range
        df_copy = df_copy.loc[subset[0]: subset[1]]
    
    if years:
        # Filtering by the specified years
        df_copy = df_copy[df_copy.index.year.isin(years)]
    
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

    Examples
    --------
    Extraction of a Daily Aggregate

    >>> from postprocessinglib.evaluation import data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> merged_df = DATAFRAMES["DF"]
    >>> print(merged_df)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980-12-31           10.20       2.530770            NaN       1.006860
    1981-01-01            9.85       2.518999            NaN       1.001954
    1981-01-02           10.20       2.507289            NaN       0.997078
    1981-01-03           10.00       2.495637            NaN       0.992233
    1981-01-04           10.10       2.484073            NaN       0.987417
    ...                    ...            ...             ...            ...
    2017-12-27             NaN       4.418050            NaN       1.380227
    2017-12-28             NaN       4.393084            NaN       1.372171
    2017-12-29             NaN       4.368303            NaN       1.364174
    2017-12-30             NaN       4.343699            NaN       1.356237
    2017-12-31             NaN       4.319275            NaN       1.348359
    
    >>> # Extract the daily aggregate by mean(default aggregation method)
    >>> daily_agg = data.daily_aggregate(df=merged_df)
    >>> print(daily_agg)
                  QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
        1980/366           10.20       2.530770             NaN       1.006860
        1981/001            9.85       2.518999             NaN       1.001954
        1981/002           10.20       2.507289             NaN       0.997078
        1981/003           10.00       2.495637             NaN       0.992233
        1981/004           10.10       2.484073             NaN       0.987417
        ...                  ...            ...             ...            ...
        2017/361             NaN       4.418050             NaN       1.380227
        2017/362             NaN       4.393084             NaN       1.372171
        2017/363             NaN       4.368303             NaN       1.364174
        2017/364             NaN       4.343699             NaN       1.356237
        2017/365             NaN       4.319275             NaN       1.348359

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-data-manipulation.ipynb>`_

    """
    # Check that there is a chosen method else return error
    if not method:
        raise RuntimeError("ERROR: A method of aggregation is required")
    else:
        # Making a copy to avoid changing the original df
        df_copy = df.copy()

        if method == "sum":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%j")).sum()
        if method == "mean":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%j")).mean()
        if method == "median":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%j")).median()
        if method == "min":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%j")).min()
        if method == "max":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%j")).max()
        if method == "inst":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%j")).last() 
    
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

    Examples
    --------
    Extraction of a Weekly Aggregate

    >>> from postprocessinglib.evaluation import data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> merged_df = DATAFRAMES["DF"]
    >>> print(merged_df)
                    QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
        1980-12-31           10.20       2.530770            NaN       1.006860
        1981-01-01            9.85       2.518999            NaN       1.001954
        1981-01-02           10.20       2.507289            NaN       0.997078
        1981-01-03           10.00       2.495637            NaN       0.992233
        1981-01-04           10.10       2.484073            NaN       0.987417
        ...                   ...            ...             ...            ...
        2017-12-27             NaN       4.418050            NaN       1.380227
        2017-12-28             NaN       4.393084            NaN       1.372171
        2017-12-29             NaN       4.368303            NaN       1.364174
        2017-12-30             NaN       4.343699            NaN       1.356237
        2017-12-31             NaN       4.319275            NaN       1.348359
    
    >>> # Extract the weekly aggregate by taking the minumum value per week
    >>> weekly_agg = data.weekly_aggregate(df=merged_df, method="min")
    >>> print(weekly_agg)
                 QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
        1980.52           10.20       2.530770             NaN       1.006860
        1981.00            9.85       2.495637             NaN       0.992233
        1981.01            8.70       2.416050             NaN       0.959137
        1981.02            8.24       2.339655             NaN       0.927429
        1981.03            7.86       2.266305             NaN       0.897038
        ...                 ...            ...             ...            ...
        2017.49             NaN       4.900197             NaN       1.536146
        2017.50             NaN       4.705044             NaN       1.472965
        2017.51             NaN       4.519744             NaN       1.413064
        2017.52             NaN       4.343699             NaN       1.356237
        2017.53             NaN       4.319275             NaN       1.348359

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-data-manipulation.ipynb>`_

    """
    # Check that there is a chosen method else return error
    if not method:
        raise RuntimeError("ERROR: A method of aggregation is required")
    else:
        # Making a copy to avoid changing the original df
        df_copy = df.copy()

        if method == "sum":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y.%U")).sum()
        if method == "mean":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y.%U")).mean()
        if method == "median":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y.%U")).median()
        if method == "min":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y.%U")).min()
        if method == "max":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y.%U")).max()
        if method == "inst":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y.%U")).last()    
    
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

    Examples
    --------
    Extraction of a Monthly Aggregate

    >>> from postprocessinglib.evaluation import data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> merged_df = DATAFRAMES["DF"]
    >>> print(merged_df)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980-12-31           10.20       2.530770            NaN       1.006860
    1981-01-01            9.85       2.518999            NaN       1.001954
    1981-01-02           10.20       2.507289            NaN       0.997078
    1981-01-03           10.00       2.495637            NaN       0.992233
    1981-01-04           10.10       2.484073            NaN       0.987417
    ...                    ...            ...             ...            ...
    2017-12-27             NaN       4.418050            NaN       1.380227
    2017-12-28             NaN       4.393084            NaN       1.372171
    2017-12-29             NaN       4.368303            NaN       1.364174
    2017-12-30             NaN       4.343699            NaN       1.356237
    2017-12-31             NaN       4.319275            NaN       1.348359
    
    >>> # Extract the monthly aggregate by taking the instantaenous value of each month
    >>> monthly_agg = data.monthly_aggregate(df=merged_df, method="inst")
    >>> print(monthly_agg)
             QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980-12           10.20       2.530770            NaN       1.006860
    1981-01            8.62       2.195846            NaN       0.867900
    1981-02            7.20       1.940355            NaN       0.762678
    1981-03            7.25       1.699932            NaN       0.664341
    1981-04           15.30       3.859564            NaN       0.584523
    ...                 ...            ...             ...            ...
    2017-08             NaN      31.050230            NaN      17.012710
    2017-09             NaN      16.144130            NaN      11.127440
    2017-10             NaN       6.123822            NaN       1.938875
    2017-11             NaN       5.164804            NaN       1.621027
    2017-12             NaN       4.319275            NaN       1.348359

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-data-manipulation.ipynb>`_

    """
    # Check that there is a chosen method else return error
    if not method:
        raise RuntimeError("ERROR: A method of aggregation is required")
    else:
        # Making a copy to avoid changing the original df
        df_copy = df.copy()

        if method == "sum":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y-%m")).sum()
        if method == "mean":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y-%m")).mean()
        if method == "median":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y-%m")).median()
        if method == "min":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y-%m")).min()
        if method == "max":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y-%m")).max()
        if method == "inst":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y-%m")).last()    
    
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
            The new dataframe with the values aggregated yearly 

    Examples
    --------
    Extraction of a Yearly Aggregate

    >>> from postprocessinglib.evaluation import data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> merged_df = DATAFRAMES["DF"]
    >>> print(merged_df)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980-12-31           10.20       2.530770            NaN       1.006860
    1981-01-01            9.85       2.518999            NaN       1.001954
    1981-01-02           10.20       2.507289            NaN       0.997078
    1981-01-03           10.00       2.495637            NaN       0.992233
    1981-01-04           10.10       2.484073            NaN       0.987417
    ...                    ...            ...             ...            ...
    2017-12-27             NaN       4.418050            NaN       1.380227
    2017-12-28             NaN       4.393084            NaN       1.372171
    2017-12-29             NaN       4.368303            NaN       1.364174
    2017-12-30             NaN       4.343699            NaN       1.356237
    2017-12-31             NaN       4.319275            NaN       1.348359
    
    >>> # Extract the yearly aggregate by taking the sum of the entire year's values
    >>> yearly_agg = data.yearly_aggregate(df=merged_df, method="sum")
    >>> print(yearly_agg)
              QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
        1980           10.20       2.530770            0.00       1.006860
        1981        10386.27    9273.383180         2424.26    4007.949313
        1982        12635.47    8874.369067         3163.23    4123.606233
        1983        11909.23    8214.793557         3198.17    3810.515038
        1984        13298.33    7459.351671         2699.42    3431.981225
        1985        13730.50    8487.241498         2992.40    3756.822014
        1986        12576.84   10651.883689         3103.15    4794.825198
        1987        15066.57    8947.025052         3599.74    4260.917801
        1988        12642.53   10377.241643         2972.87    4614.234614
        1989        10860.93   11118.336160         2624.79    5193.322199
        1990        11129.76   11226.011936         2650.50    5273.448490
        1991        14354.61   12143.013205         3215.89    5732.371571
        1992        17033.16    9919.064629         3885.72    4566.044810
        1993        15238.65   10265.868953         3598.67    4700.055333
        1994        15623.13    8064.390172         3777.16    4053.331783
        1995        12892.89   10526.186570         3817.08    5006.592916
        1996        12551.39    9191.247302         3249.36    4195.638177
        1997           11.20    9078.253847            0.00    4469.825844
        1998            0.00    9421.178402         3598.21    4650.819283
        1999            0.00    8683.319193         3220.62    4032.381482
        2000            0.00   10181.718825            0.00    4921.033689
        2001            0.00    7076.942619            0.00    3525.593143
        2002            0.00    8046.998223            0.00    4048.992212
        2003            0.00    9017.711719            0.00    4517.088194
        2004            0.00   11726.707770            0.00    4941.582065
        2005            0.00   11975.002047            0.00    4700.295391
        2006            0.00    8972.956022            0.00    4038.214837
        2007            0.00   11089.242586            0.00    5035.426223
        2008            0.00    9652.958064            0.00    4630.531909
        2009            0.00    8762.313253            0.00    3659.265122
        2010            0.00    8006.621137            0.00    3475.115315
        2011            0.00   10158.521707            0.00    4748.153725
        2012            0.00   13141.668859            0.00    5847.670810
        2013            0.00   11389.072535            0.00    4769.917090
        2014            0.00   12719.851800            0.00    5298.904086
        2015            0.00   12258.178724            0.00    5362.497143
        2016            0.00    9989.779678            0.00    4269.909376
        2017            0.00    8801.897128            0.00    4226.258100

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-data-manipulation.ipynb>`_

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


def generate_dataframes(csv_fpath: str='', sim_fpath: str='', obs_fpath: str='', warm_up: int = 0, start_date :str = "", end_date: str = "",
                        daily_agg:bool=False, da_method:str="", weekly_agg:bool=False, wa_method:str="",
                        monthly_agg:bool=False, ma_method:str="", yearly_agg:bool=False, ya_method:str="",
                        seasonal_p:bool=False, sp_dperiod:tuple[str, str]=[], sp_subset:tuple[str, str]=None,
                        long_term:bool=False, lt_method=None) -> dict[str, pd.DataFrame]:
    """ 
    Function to Generate the required dataframes

    Parameters
    ----------
    csv_fpath : string
            the path to the csv file. It can be relative or absolute. If given, sim_fpath and obs_fpath
            must be None.
    sim_fpath: str
        The filepath to the simulated csv of data. If given obs_fpath must also be given and csv_fpath
        must be None. 
    obs_fpath: str
        The filepath to the observed csv of the data. If given sim_fpath must also be given and csv_fpath
        must be None.
    warm_up: int 
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
    sp_subset: tuple(str, str)
            A tuple of string values representing the start and end dates of the time range. Format is YYYY-MM-DD.
    longterm: bool = False
            If True calculates the min, max and median values for the long term seasonal. It will also create
            additional dataframes depending on the value of 'lt_method'.
    lt_method
            Specifies extra long term dataframes to create

    Returns
    -------
    dict[str, pd.dataframe]
            A dictionary containing each Dataframe requested. Its default content is:

            - DF = merged dataframe
            - DF_SIMULATED = all simulated data
            - DF_OBSERVED = all observed data
            
            Depending on which you requested it can also contain:

            - DF_DAILY = dataframe aggregated by days of the year
            - DF_WEEKLY = dataframe aggregated by the weeks of the year
            - DF_MONTHLY = dataframe aggregated by months of the year
            - DF_YEARLY = dataframe aggregated by all the years in the data
            - DF_CUSTOM = dataframe truncated as per the seasonal period parameters
            - DF_LONGTERM_MIN = long term seasonal dataframe aggregated using the min of its daily values 
            - DF_LONGTERM_MAX = long term seasonal dataframe aggregated using the max of its daily values
            - DF_LONGTERM_MEAN =  long term seasonal dataframe aggregated using the mean of its daily values
              
              Depending on "lt_method," you can also request that it contain:

                - DF_LONGTERM_SUM = long term seasonal dataframe aggregated using the sum of its daily values
                - DF_LONGTERM_MEDIAN = long term seasonal dataframe aggregated using the median of its daily values
                - DF_LONGTERM_Q1 = long term seasonal dataframe aggregated showing the first quartile of its daily
                - DF_LONGTERM_Q2 = long term seasonal dataframe aggregated showing the second quartile of its daily
                - DF_LONGTERM_Q3 = long term seasonal dataframe aggregated showing the third quartile of its daily
         

    Example
    -------
    See linked jupyter `notebook <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-data-manipulation.ipynb>`_ file for usage instances
            
    """

    DATAFRAMES = {}
    if csv_fpath:
        # read the combined csv file into a dataframe
        df = pd.read_csv(csv_fpath, skipinitialspace = True, index_col = [0, 1])
        # if there are any extra columns at the end of the csv file, remove them
        if len(df.columns) % 2 != 0:
            df.drop(columns=df.columns[-1], inplace = True)        
        # Convert the year and jday index to datetime indexing
        start_day = hlp.MultiIndex_to_datetime(df.index[0])
        df.index = pd.to_datetime([i for i in range(len(df.index))], unit='D',origin=pd.Timestamp(start_day))
        # replace all invalid values with NaN
        df = df.replace([-1, 0], np.nan)   
        
        # Take off the warm up time
        DATAFRAMES["DF"] = df[warm_up:]    
        observed = df[warm_up:].iloc[:, ::2] 
        simulated = df[warm_up:].iloc[:, 1::2]

    elif sim_fpath and obs_fpath:
        # read the simulated and observed csv files into dataframes
        sim_df = pd.read_csv(sim_fpath, skipinitialspace = True, index_col=[0, 1])
        obs_df = pd.read_csv(obs_fpath, skipinitialspace = True, index_col=[0, 1])

        # Convert the year and jday index to datetime indexing
        # simulated
        start_day = hlp.MultiIndex_to_datetime(sim_df.index[0])
        sim_df.index = pd.to_datetime([i for i in range(len(sim_df.index))], unit='D',origin=pd.Timestamp(start_day))
        
        
        # observed
        start_day = hlp.MultiIndex_to_datetime(obs_df.index[0])
        obs_df.index = pd.to_datetime([i for i in range(len(obs_df.index))], unit='D',origin=pd.Timestamp(start_day))

        # replace all invalid values with NaN
        sim_df = sim_df.replace([-1, 0], np.nan)
        obs_df = obs_df.replace([-1, 0], np.nan)
        df = pd.DataFrame(index = obs_df.index)
        for j in range(0, len(obs_df.columns)):
            arr1 = obs_df.iloc[:, j]
            arr2 = sim_df.iloc[:, j]
            df = pd.concat([df, arr1, arr2], axis = 1)

        # Take off the warm up time
        simulated = sim_df[warm_up:]
        observed = obs_df[warm_up:]                
        DATAFRAMES["DF"] = df[warm_up:] 

    else:
        raise RuntimeError('either sim_fpath and obs_fpath or csv_fpath are required inputs.')
       

    # splice the dataframes according to the time frame
    if not start_date and end_date:
        # there's an end date but no start date
        simulated = simulated.loc[:end_date]
        observed = observed.loc[:end_date]
        DATAFRAMES["DF"] = DATAFRAMES["DF"][:end_date]
    elif not end_date and start_date:
        # there's and end date but no start date
        simulated = simulated.loc[start_date:]
        observed = observed.loc[start_date:]
        DATAFRAMES["DF"] = DATAFRAMES["DF"][start_date:]
    elif start_date and end_date:
        # there's a start and end date
        simulated = simulated.loc[start_date:end_date]
        observed = observed.loc[start_date:end_date]
        DATAFRAMES["DF"] = DATAFRAMES["DF"][start_date:end_date]

    print(f"The start date for the Observed Data is {observed.index[0].strftime('%Y-%m-%d')}")
    print(f"The start date for the Simulated Data is {simulated.index[0].strftime('%Y-%m-%d')}")
    print(f"The start date for the Merged Data is {DATAFRAMES['DF'].index[0].strftime('%Y-%m-%d')}")
    
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
    elif seasonal_p and sp_dperiod and sp_subset:
        DATAFRAMES["DF_CUSTOM"] = seasonal_period(df = DATAFRAMES["DF"], daily_period=sp_dperiod,
                                                  subset=sp_subset)    
    elif seasonal_p and sp_dperiod:
        DATAFRAMES["DF_CUSTOM"] = seasonal_period(df = DATAFRAMES["DF"], daily_period=sp_dperiod)

    # 6. long term seasonal
    if long_term:
        DATAFRAMES["LONG_TERM_MIN"] = long_term_seasonal(df=DATAFRAMES["DF"], method="min")
        DATAFRAMES["LONG_TERM_MAX"] = long_term_seasonal(df=DATAFRAMES["DF"], method="max")
        DATAFRAMES["LONG_TERM_MEDIAN"] = long_term_seasonal(df=DATAFRAMES["DF"], method="median")
        if lt_method is None:
            lt_method = []
        elif isinstance(lt_method, str):
            lt_method = [lt_method]
        elif not isinstance(lt_method, list):
            raise ValueError("Argument must be a string or a list of strings.")

        for method in lt_method:
            if not isinstance(method, str):
                raise ValueError("All items in the list must be strings.")    
            DATAFRAMES[f"LONG_TERM_{method.upper()}"] = long_term_seasonal(df=DATAFRAMES["DF"], method=method)
    
    
    return DATAFRAMES
