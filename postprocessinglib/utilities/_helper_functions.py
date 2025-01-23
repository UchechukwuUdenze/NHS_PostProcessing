"""
The helper module contains all of the functions used alongside the metrics to filter, limit, validate the data 
before it gets evaluated and present the data properly

"""

import numpy as np
import pandas as pd
from datetime import datetime
from math import floor, log10

from postprocessinglib.utilities._errors import AllInvalidError

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

    Parameters:
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
        an index representing the year and jday index of the dataframe
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

def MultiIndex_to_datetime(index: tuple) -> str:
    """ Convert the MultiIndex value to a datetime value for use in the dataframe

    Parameters:
    -----------
    index: tuple[int, int]
            an index representig the year and jday index of the dataframe

    Returns:
    --------
    str:
        a string of the date in the format "yyyy-mm-dd"
    """
    year = str(index[0])
    jday = str(index[1])
     
    # adjusting day num
    jday.rjust(3 + len(jday), '0')
     
    # converting to date
    res = datetime.strptime(year + "-" + jday, "%Y-%j").strftime("%Y-%m-%d")
     
    # printing result
    return str(res)

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

    if (len(observed.shape) < 1) or (observed.shape[1] < 1):
        raise RuntimeError("observed or simulated data is incomplete")


def filter_valid_data(df: pd.DataFrame, station_num: int = 0, station: str = "") -> pd.DataFrame:
    """ Removes the invalid values from a dataframe

    Parameters
    ----------
    df : pd.DataFrame
            the dataframe which you want to remove invalid values from
    station_num : int
            the number referring to the station values we are trying to modify
    station: str = ""
            the column name representing the station from which you want to 
            remove invalid values from

    Returns
    -------
    pd.DataFrame: 
            the modified input dataframe with rows containing NaN, zero, negative
            and inf values removed.
    """

    if not station and station_num < 0 and station_num >= df.shape[1] :
        raise ValueError("You must have either a valid station number or station name")                                                                            
    
    # Replaces infinities with zeros
    df = df.replace([np.inf, -np.inf], np.nan)
    # Replaces nan with zeros
    df = df.replace(np.nan, 0)
    
    if not station:
        # drop zeros and negatives
        df = df.drop(df[df.iloc[:, station_num] <= 0.0].index)                
        return df
    
    # drop zeros and negatives
    df = df.drop(df[df[station] <= 0.0].index)
    return df        


def sig_figs(x: float, precision: int)-> float:
    """
    Rounds a number to number of significant figures

    Parameters
    ----------
    x: float
        the number to be rounded
    precision: int
        the number of significant figures

    Returns
    -------
    float:
        the number rounded to the requested significant figures
    """
    if not np.isnan(x):
        x = float(x)
        precision = int(precision)
        return round(x, -int(floor(log10(abs(x)))) + (precision - 1))
    else:
        return np.nan
    