"""
The preprocessing module contains all of the functions used alongside the metrics to filter, limit and validate the data 
before it gets evaluated.

"""

import numpy as np
import pandas as pd

from postprocessinglib.utilities.errors import AllInvalidError

def check_valid_dataframe(observed: pd.DataFrame, simulated: pd.DataFrame):
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
    
    
def is_leap_year(year: int) -> bool:
    """ Determines if a year is a leap year

    Paremeters:
    -----------
    year: int
            the year being checked

    Returns:
    --------
    bool: 
        True if it is a leap year, False otherwise
    """
    if year % 4 == 0:
        return True
    return False

def datetime_to_index(datetime :str)-> tuple[int, int]:
    """ Convert the datetime value to index value for use in the dataframe

    Parameters:
    -----------
    datetime: str
            a string containing the date being searched for entered in the format "yyyy-mm-dd"

    Returns:
    --------
    tuple: [int, int]
        an index representig the year and jday index of the dataframe
    """
    year, month, day = datetime.split("-")
    jday = 0
    for i in range(1, int(month)):
        if i == 1 or i == 3 or i == 5 or i == 7 or i == 8 or i == 10 or i == 12:
            # the months with 31 days
            jday += 31
        elif i == 4 or i == 6 or i == 9 or i == 11:
            # the months with 30 days
            jday += 30
        else:     #i == 2 (february)
            if is_leap_year(int(year)):
                jday += 29
            else:
                jday += 28

    jday += int(day)        
    return(int(year), jday)


def validate_data(observed: pd.DataFrame, simulated: pd.DataFrame):
    """ Ensures that a set of observed and simulated dataframes are valid

    Raises
    ------
    RuntimeError:
            if the sizes or shapes of both dataframes are not the same
    """
    if not isinstance(observed, pd.DataFrame) or not isinstance(simulated, pd.DataFrame):
        raise ValueError("Both observed and simulated values must be pandas DataFrames.")
    
    if observed.shape != simulated.shape:
        raise RuntimeError("Shapes of observations and simulations must match")

    if (len(observed.shape) < 2) or (observed.shape[1] < 2):
        raise RuntimeError("observed or simulated data is incomplete")


def filter_valid_data(df: pd.DataFrame, station_num: int = 0, station: str = "",
                  remove_neg: bool = False, remove_zero: bool = False, remove_NaN: bool = False,
                  remove_inf: bool = False) -> pd.DataFrame:
    """ Removes the invalid values from a dataframe

    Parameters
    ----------
    df : pd.DataFrame
            the dataframe which you want to remove invalid values from
    station_num : int
            the number referring to the station values we are trying to modify
    remove_neg = True: bool 
            indicates that the negative fields are the inavlid ones
    remove_zero = True: bool 
            indicate that the zero fields are the invalid ones
    remove_NaN = True: bool 
            indicate that the empty fields are the invalid ones
    remove_inf = True: bool
            indicate that the inf fields are the invalid ones

    Returns
    -------
    pd.DataFrame: 
            the modified input dataframe 

    """

    if not station and station_num < 0 and station_num >= df.shape[1] :
        raise ValueError("You must have either a valid station number or station name")
    
    if not station:
        if remove_neg:
            df = df.drop(df[df.iloc[:, station_num] <= 0.0].index)
        elif remove_zero:
            df = df.drop(df[df.iloc[:, station_num] == 0.0].index)
        elif remove_NaN:
            df = df.drop(df[df.iloc[:, station_num] == np.nan].index)
        elif remove_inf:
            df = df.drop(df[df.iloc[:, station_num] == np.inf].index)        
        return df

    if remove_neg:
        df = df.drop(df[df[station] < 0.0].index)
    elif remove_zero:
        df = df.drop(df[df[station] == 0.0].index)
    elif remove_NaN:
        df = df.drop(df[df[station] == np.nan].index)
    elif remove_inf:
        df = df.drop(df[df[station] == np.inf].index)        
    return df