"""
The helper module contains all of the functions used alongside the metrics to filter, limit, validate the data 
before it gets evaluated and present the data properly. 

It uses ``check_valid_dataframe`` to check if a dataframe just contains invalid values such as Nan or negative values.
It contains functions like ``filer_valid_data`` which help in filtering out rows contain Nan, negative or zero values.
It also contains functions like ``validate_data`` which help in making sure that the dataframes are valid and have the 
same shape and size. Functions like ``sig_figs`` and ``leap_year``which help in rounding numbers
to a certain number of significant figures and determining if a year is a leap year respectively

"""

import numpy as np
import pandas as pd
from datetime import datetime
import math as math

from postprocessinglib.utilities._errors import AllInvalidError

def check_valid_dataframe(observed: pd.DataFrame, simulated: pd.DataFrame):
    """Check if all observations or simulations are invalid and raise an exception/error if this is the case.

        Invalid in this case refers to all values in the dataframe being NaN, negative or zero. It goes through 
        both the observed and simulated seperately and makes sure that its values are not all Nan, negative or zero.
        If any of these conditions are not met, it raises an AllInvalidError specifying which condition was not met.  

    Parameters:
    -----------
    observed: pd.DataFrame
            The observed dataframe being checked.

    simulated: pd.DataFrame
            The simulated dataframe being checked.

    Raises
    ------
    AllInvalidError
        If all observations or all simulations are NaN or negative.

    Example:
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.utilities import _helper_functions
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = data, columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> obs = test_df.iloc[:, ::2]
    >>> sim = test_df.iloc[:, 1::2]
    >>> print(test_df)
            obs1     sim1     obs2     sim2
    1981    NaN     -inf       -1     -inf
    1982    NaN      inf      NaN      NaN
    1983    inf     -inf      NaN     -inf
    1984   -inf      NaN      inf      NaN
    1985    NaN     -inf     -inf      inf
    1986   -inf      NaN      NaN      inf
    1987    NaN     -inf      inf      NaN
    1988    inf     -inf      NaN      NaN
    1989    inf      inf      NaN     -inf
    1990   -inf      inf      NaN     -inf
    >>> .
    >>> _helper_functions.check_valid_dataframe(observed=obs, simulated=sim)
    >>> # Error is raised due to all the observed values being invalid
    >>> # The error message is as follows: "All observed values are NaN"
    """
    if len(observed.index) == 0:
        raise AllInvalidError("All observed values are NaN")
    if len(simulated.index) == 0:
        raise AllInvalidError("All simulated values are NaN")
    if (observed.values < 0).any():
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

    Example:
    --------
    Convert the follwing datetime values to index values: "2020-01-01", "1995-12-31", "2000-02-29"

    >>> datetime_to_index("2020-01-01")
        (2020, 1)
    >>> datetime_to_index("1995-12-31")
        (1995, 365)
    >>> datetime_to_index("2000-02-29")
        (2000, 60)

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

    Example:
    --------
    Convert the follwing multiindex values to datetime values: (2020, 1), (1995, 365), (2000, 60)

    >>> MultiIndex_to_datetime((2020, 1))
        "2020-01-01"
    >>> MultiIndex_to_datetime((1995, 365))
        "1995-12-31"
    >>> MultiIndex_to_datetime((2000, 60))
        "2000-02-29"

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

        Invalid in this case refers to the inputs not being dataframes, both Dataframes not having the same 
        shape and size or we having dataframes with no data (empty dataframes). It goes through both the observed 
        and simulated dataframes, comparing them where necessary, and making sure that the above conditions are met.
        If any of these conditions are not met, it raises the corresponding error.

    Parameters:
    -----------
    observed: pd.DataFrame
            The observed dataframe being checked.

    simulated: pd.DataFrame
            The simulated dataframe being checked.

    Raises
    ------
    RuntimeError:
            if the sizes or shapes of both dataframes are not the same 
            or if the dataframes are empty

    ValueError:
            if the inputs are not dataframes 

    Example:
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.utilities import _helper_functions
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = data, columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> obs = test_df.iloc[:, ::2]
    >>> sim_test_1 = test_df.iloc[:, 1]
    >>> sim_test_2 = np.array([1, 2, 3, 4])
    >>> sim_test_3 = pd.DataFrame(index = index)
    >>> print(obs)
              obs1      obs2
    1981       NaN  0.761252
    1982  0.332201  0.229883
    1983  0.251259       inf
    1984  0.620732  0.266127
    1985      -inf      -inf
    1986  0.013643  0.473920
    1987       NaN  0.466392
    1988  0.115222       NaN
    1989  0.434341       NaN
    1990       NaN  0.428369
    >>> print(sim_test_1)
              sim1
    1981       inf
    1982  0.577598
    1983      -inf
    1984  0.121188
    1985  0.038524
    1986  0.992656
    1987  0.422545
    1988  0.319578
    1989       inf
    1990  0.658737
    >>> print(sim_test_2)
    [1 2 3 4]
    >>> print(sim_test_3)
    Empty DataFrame
    Columns: []
    Index: [1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990]
    >>> . 
    >>> _helper_functions.validate_data(observed=obs, simulated=sim_test_1)
    >>> # Error is raised due to the different shapes of the dataframes
    >>> # The error message is as follows: "Shapes of observations and simulations must match"
    >>> .
    >>> _helper_functions.validate_data(observed=obs, simulated=sim_test_2)
    >>> # Error is raised due to the simulated data not being a dataframe
    >>> # The error message is as follows: "Both observed and simulated values must be pandas DataFrames."
    >>> .
    >>> _helper_functions.validate_data(observed=obs, simulated=sim_test_3)
    >>> # Error is raised due to the simulated data being an empty dataframe
    >>> # The error message is as follows: "observed or simulated data is incomplete"

    """
    if not isinstance(observed, pd.DataFrame) or not isinstance(simulated, pd.DataFrame):
        raise ValueError("Both observed and simulated values must be pandas DataFrames.")
    
    if observed.shape != simulated.shape:
        raise RuntimeError("Shapes of observations and simulations must match")

    if (len(observed.shape) < 1) or (observed.shape[1] < 1) or (len(simulated.shape) < 1) or (simulated.shape[1] < 1):
        raise RuntimeError("observed or simulated data is incomplete")


def filter_valid_data(df: pd.DataFrame, station_num: int = 0, station: str = "") -> pd.DataFrame:
    """ Removes the invalid values from a dataframe
    
        Invalid in this case refers to  NaN, negative and infinity. It goes through the dataframe, checking 
        the individual colummn indentified by station_num or station for whether it contains Nan, negative or
        infinity and removes the rows that contain these values. 

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
            the modified input dataframe with rows containing NaN, negative
            and inf values removed.
    
    Example:
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.utilities import _helper_functions
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = data, columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
            obs1      sim1      obs2      sim2
    1981       NaN       inf  0.761252       NaN
    1982  0.332201  0.577598  0.229883      -inf
    1983  0.251259      -inf       inf  0.006224
    1984  0.620732  0.121188  0.266127  0.324645
    1985      -inf  0.038524      -inf  0.126910
    1986  0.013643  0.992656  0.473920  0.536638
    1987       NaN  0.422545  0.466392  0.363516
    1988  0.115222  0.319578       NaN  0.181709
    1989  0.434341       inf       NaN  0.971740
    1990       NaN  0.658737  0.428369       NaN
    >>> .
    >>> valid_data = _helper_functions.filter_valid_data(df=test_df, station_num=0) 
    >>> print(valid_data) 
            obs1      sim1      obs2      sim2
    1982  0.332201  0.577598  0.229883  0.000000
    1983  0.251259  0.000000  0.000000  0.006224
    1984  0.620732  0.121188  0.266127  0.324645
    1986  0.013643  0.992656  0.473920  0.536638
    1988  0.115222  0.319578  0.000000  0.181709
    1989  0.434341  0.000000  0.000000  0.971740
    >>> ## Obsereve how all the 'invalid' values in obs1 have been removed.

    """

    if not station and station_num < 0 and station_num >= df.shape[1] :
        raise ValueError("You must have either a valid station number or station name")                                                                            
    
    # Replaces infinities with zeros
    df = df.replace([np.inf, -np.inf], np.nan)
    # Replaces nan with zeros
    df = df.replace(np.nan, -1)
    
    if not station:
        # drop zeros and negatives
        df = df.drop(df[df.iloc[:, station_num] < 0.0].index)                
        return df
    
    # drop zeros and negatives
    df = df.drop(df[df[station] < 0.0].index)
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
        return round(x, -int(math.floor(math.log10(abs(x)))) + (precision - 1))
    else:
        return np.nan
    