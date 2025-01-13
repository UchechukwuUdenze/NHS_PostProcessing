"""
The visual module contains different plotting functions for time series visualization.
It allows users to plot hydrographs per station for each stations to allow us visualize
the time-series data

Some of them also allow their metrics to be placed beside the plots

"""

from typing import Union
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from shapely.geometry import Point

from postprocessinglib.evaluation import metrics
from postprocessinglib.utilities import helper_functions as hlp

def plot(merged_df: pd.DataFrame = None, df: pd.DataFrame = None, obs_df: pd.DataFrame = None, sim_df: pd.DataFrame = None,
         legend: tuple[str, str] = ('Simulated Data', 'Observed Data'), metrices: list[str] = None,
         grid: bool = False, title: str = None, labels: tuple[str, str] = None, padding: bool = False ,
         linestyles: tuple[str, str] = ('r-', 'b-'), linewidth: tuple[float, float] = (1.5, 1.25),
         fig_size: tuple[float, float] = (10,6), metrics_adjust: tuple[float, float] = (1.05, 0.5),
         plot_adjust: float = 0.2, save: bool=False, save_as:str = None, directory:str = ".") ->plt.figure:
    """ Create a comparison time series line plot of simulated and observed time series data

    Parameters
    ----------
    merged_df : pd.DataFrame
        the dataframe containing the series of observed and simulated values. It must have a datetime
        index and an even number of columns where in any two, the left column is the Measured/observed
        data and the right is the Simulated data. If it is present, the obs_df and sim_df must be None.
    
    obs_df : pd.DataFrame
        A DataFrame conataning rows of measured data. It must have a datetime index. if it is
        present it is accompanied by the sim_df and the merged_df must be None.

    sim_df : pd.DataFrame
        A DataFrame contaning rows of predicted/simulated data. It must have a datetime index. if it is
        present it is accompanied by the obs_df and the merged_df must be None.

    df : pd.DataFrame
        A single dataframe rows of data. It must have a datetime index. if it is present obs_df, sim_df
        and the merged_df must be None. It is to be used in cases where the user wishes to plot only
        one of either simulated or observed data.

    legend: tuple[str, str]
        Adds a Legend in the 'best' location determined by matplotlib.

    metrices: list[str]
        Adds Metrics to the right side of the plot. Any metric from the postprocessing.metrics library
        can be added to the plot as the abbreviation of the function. The entries must be in a list.
        (e.g. ['PBIAS', 'MSE', 'KGE']).

    grid: bool
        If True, adds a grid to the plot.

    title: str or list[str]
        If given, adds a titles to the plots.

    labels: tuple[str, str]
        List of two string inputs specifying x-axis labels and y-axis labels, respectively.

    padding: bool
        If true, will set the padding to zero for the lines in the line plot.
    
    linestyles: tuple[str, str]
        List of two string inputs that will change the linestyle of the simulated and
        recorded data, respectively.

    linewidth: tuple[float, float]
        Tuple of length two that specifies the thickness of the lines for both the Simulated and 
        recorded data, respectively

    fig_size: tuple[float, float]
        Tuple of length two that specifies the horizontal and vertical lengths of the plot in
        inches, respectively.

    metrics_adjust: tuple[float, float]
        Tuple of length two indicating the relative position of the text (x-coordinate, y-coordinate)
        when adding metrics to the plot.

    plot_adjust: float
        Specifies the relative position to shift the plot to the left when adding metrics to the
        plot.
    
    save: bool
        If True, the plot images will be saved as png files in the format plot_1.png, plot_2.png,
        etc., depending how many plots are generated or with the names specified in the save_as 
        variable
    
    save_as: str or list[str]
        If provided these wil be the naming nomenclature used to save the figures as specified by the
        save variable 
    
    directory: str
        The directory where the plots will be saved. The current directory will be searched
        for the spedified 'directory' directory, if none is found, it will be created. Its default is the
        current directory represented by '.' .

    Returns
    -------
    fig : Matplotlib figure instance and/or png files of the figures.
    
    Examples
    --------
    Visualization of a set of data from two stations using a 2D line plot

    >>> from postprocessinglib.evaluation import metrics, visuals, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365, monthly_agg = True)
    >>> observed = DATAFRAMES["DF_OBSERVED"] 
    >>> simulated = DATAFRAMES["DF_SIMULATED"]
    >>> merged_df = DATAFRAMES["DF"]
    >>> monthly_df = DATAFRAMES["DF_MONTHLY"]
    >>> .
    >>> # plot of the stations in the dataframe from 1981 till 1990
    >>> visuals.plot(merged_df = merged_df['1981-01-01':'1990-12-31'],
                    title='Hydrograph of the daily time series from 1981-1990',
                    linestyles=['r-', 'b-'],
                    labels=['Datetime', 'Streamflow'],
                    metrices=['RMSE', 'MAE', 'KGE'],
                    linewidth = [.75, 1.25],
                    grid=True
                    )

    .. image:: ../Figures/plot_Station_1_1981_to_1990.png
    .. image:: ../Figures/plot_Station_2_1981_to_1990.png

    >>> # plot of the stations in the dataframe from 1981 till 1990 aggregated monthly by mean(default)
    >>> visuals.plot(merged_df = monthly_df['1981-01':'1990-12'],
                    title='Hydrograph of the time series aggregated monthly from 1981-1990',
                    linestyles=['r-', 'b-'],
                    labels=['Datetime', 'Streamflow'],
                    metrices=['RMSE', 'MSE', 'PBIAS'],
                    grid=True
                    )

    .. image:: ../Figures/plot_monthly_Station_1_1981_to_1990.png
    .. image:: ../Figures/plot_monthly_Station_2_1981_to_1990.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/Examples.ipynb>`_
         
    """
    if merged_df is not None and sim_df is None and obs_df is None and df is None:
        # Setting Variable for the simulated data, observed data, and time stamps
        obs = merged_df.iloc[:, [0]]
        sim = merged_df.iloc[:, [1]]
        for j in range(2, len(merged_df.columns), 2):
            obs = pd.concat([obs, merged_df.iloc[:, j]], axis = 1)
            sim = pd.concat([sim, merged_df.iloc[:, j+1]], axis = 1)
        time = merged_df.index
    elif sim_df is not None and obs_df is not None and merged_df is None and df is None:
        obs = obs_df
        sim = sim_df
        time = obs_df.index
    elif sim_df is None and obs_df is None and merged_df is None and df is not None:
        time = df.index
        line_df = df
    else:
        raise RuntimeError('either sim_df and obs_df or merged_df or df are required inputs.')

    # Convert time index to float or int if not datetime
    if not isinstance(time, pd.DatetimeIndex):
        # if the index is not a datetime, then it was converted during aggregation to
        # either a string (most likely) or an int or float
        if (isinstance(time[0], int)) or (isinstance(time[0], float)):
            pass
        else:
            if '/' in time[0]:
                # daily
                time = [pd.Timestamp(datetime.datetime.strptime(week, '%Y/%j').date()) for week in time]
            elif '.' in time[0]:
                print(True)
                # weekly
                # datetime ignores the week specifier unless theres a weekday attached,
                # so we attach Sunday - day 0
                time = [week+'.0' for week in time]
                time = [pd.Timestamp(datetime.datetime.strptime(week, '%Y.%U.%w').date()) for week in time]
            elif '-' in time[0]:
                # monthly
                time = [pd.Timestamp(datetime.datetime.strptime(week, '%Y-%m').date()) for week in time]
            else: # yearly
                time = np.asarray(time, dtype='float')
    if df is None:
        if len(obs.columns) <= 5:
            for i in range (0, len(obs.columns)):
                # Plotting the Data     
                fig = plt.figure(figsize=fig_size, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111)                       
                plt.plot(time, obs[obs.columns[i]], linestyles[1], label=legend[1], linewidth = linewidth[1])
                plt.plot(time, sim[sim.columns[i]], linestyles[0], label=legend[0], linewidth = linewidth[0])
                plt.legend(fontsize=15)

                # Adjusting the plot if user wants tight x axis limits
                if padding:
                    plt.xlim(time[0], time[-1])

                plt.xticks(fontsize=15, rotation=45)
                plt.yticks(fontsize=15)

                # Placing Labels if requested
                if labels:
                    # Plotting Labels
                    plt.xlabel(labels[0], fontsize=18)
                    plt.ylabel(labels[1]+"m\u00B3/s", fontsize=18)
                if title:
                    title_dict = {'family': 'sans-serif',
                                'color': 'black',
                                'weight': 'normal',
                                'size': 20,
                                }
                    ## Check that the title is a list of strings or a single string
                    if isinstance(title, list):
                        try:
                            if title[i] == '':
                                    ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)
                            else:
                                ax.set_title(label=title[i], fontdict=title_dict, pad=25)
                        except IndexError:
                            ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)                        
                    elif isinstance(title, str):
                        ax.set_title(label=title, fontdict=title_dict, pad=25)

                # Placing a grid if requested
                if grid:
                    plt.grid(True)

                # Fixes issues with parts of plot being cut off
                plt.tight_layout()

                # Placing Metrics on the Plot if requested
                if metrices:
                    formatted_selected_metrics = 'Metrics:\n'
                    if metrices == 'all':
                        for key, value in metrics.calculate_all_metrics(observed=obs, simulated=sim).items():
                            formatted_selected_metrics += key + ' : ' + str(value[i]) + '\n'
                    else: 
                        assert isinstance(metrices, list)
                        for key, value in metrics.calculate_metrics(observed=obs, simulated=sim, metrices=metrices).items():
                            formatted_selected_metrics += key + ' : ' + str(value[i]) + '\n'

                    font = {'family': 'sans-serif',
                            'weight': 'normal',
                            'size': 12}
                    plt.text(metrics_adjust[0], metrics_adjust[1], formatted_selected_metrics, ha='left',
                            va='center', transform=ax.transAxes, fontdict=font, #mouseover = True,
                            bbox = dict(boxstyle = "round, pad = 0.5,rounding_size=0.3", facecolor = "0.8", edgecolor="k"))

                    plt.subplots_adjust(right = 1-plot_adjust)

                # save to file if requested 
                if save:
                    # Check if the directory exists
                    if not os.path.exists(directory):
                        # If the directory does not exist, create it
                        os.makedirs(directory)
                        print(f"Directory '{directory}' created.")
                    elif directory != ".":
                        print(f"Directory '{directory}' already exists.")
                    else:
                        print("Plots will be saved to current directory")
                    ## Check that the title is a list of strings or a single string
                    if isinstance(save_as, list):
                        try:
                            if save_as[i] == '':
                                plt.savefig(os.path.join(directory, f"plot_{i+1}.png"))
                            else:
                                plt.savefig(os.path.join(directory, f"{save_as[i]}.png"))
                        except IndexError:
                            plt.savefig(os.path.join(directory, f"plot_{i+1}.png"))                        
                    elif isinstance(save_as, str):
                        plt.savefig(os.path.join(directory, f"{save_as}_{i+1}.png"))
                    else:
                        plt.savefig(os.path.join(directory, f"plot_{i+1}.png"))
        else:
            for i in range (0, len(obs.columns)):
                # Plotting the Data     
                fig = plt.figure(figsize=fig_size, facecolor = 'w', edgecolor='k')
                ax = fig.add_subplot(111)                       
                plt.plot(time, obs[obs.columns[i]], linestyles[1], label=legend[1], linewidth = linewidth[1])
                plt.plot(time, sim[sim.columns[i]], linestyles[0], label=legend[0], linewidth = linewidth[0])
                plt.legend(fontsize=15)

                # Adjusting the plot if user wants tight x axis limits
                if padding:
                    plt.xlim(time[0], time[-1])

                plt.xticks(fontsize=15, rotation=45)
                plt.yticks(fontsize=15)

                # Placing Labels if requested
                if labels:
                    # Plotting Labels
                    plt.xlabel(labels[0], fontsize=18)
                    plt.ylabel(labels[1]+"m\u00B3/s", fontsize=18)
                if title:
                    title_dict = {'family': 'sans-serif',
                                'color': 'black',
                                'weight': 'normal',
                                'size': 20,
                                }
                    ## Check that the title is a list of strings or a single string
                    if isinstance(title, list):
                        try:
                            if title[i] == '':
                                    ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)
                            else:
                                ax.set_title(label=title[i], fontdict=title_dict, pad=25)
                        except IndexError:
                            ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)                        
                    elif isinstance(title, str):
                        ax.set_title(label=title, fontdict=title_dict, pad=25)

                # Placing a grid if requested
                if grid:
                    plt.grid(True)

                # Fixes issues with parts of plot being cut off
                plt.tight_layout()

                # Placing Metrics on the Plot if requested
                if metrices:
                    formatted_selected_metrics = 'Metrics:\n'
                    if metrices == 'all':
                        for key, value in metrics.calculate_all_metrics(observed=obs, simulated=sim).items():
                            formatted_selected_metrics += key + ' : ' + str(value[i]) + '\n'
                    else: 
                        assert isinstance(metrices, list)
                        for key, value in metrics.calculate_metrics(observed=obs, simulated=sim, metrices=metrices).items():
                            formatted_selected_metrics += key + ' : ' + str(value[i]) + '\n'

                    font = {'family': 'sans-serif',
                            'weight': 'normal',
                            'size': 12}
                    plt.text(metrics_adjust[0], metrics_adjust[1], formatted_selected_metrics, ha='left',
                            va='center', transform=ax.transAxes, fontdict=font, #mouseover = True,
                            bbox = dict(boxstyle = "round, pad = 0.5,rounding_size=0.3", facecolor = "0.8", edgecolor="k"))

                    plt.subplots_adjust(right = 1-plot_adjust)
                
                # Check if the directory exists
                if not os.path.exists(directory):
                    # If the directory does not exist, create it
                    os.makedirs(directory)
                    print(f"Directory '{directory}' created.")
                elif directory != ".":
                    print(f"Directory '{directory}' already exists.")
                else:
                    print("Plots will be saved to current directory")
                ## Check that the title is a list of strings or a single string
                if isinstance(save_as, list):
                    try:
                        if save_as[i] == '':
                            plt.savefig(os.path.join(directory, f"plot_{i+1}.png"))
                        else:
                            plt.savefig(os.path.join(directory, f"{save_as[i]}.png"))
                    except IndexError:
                        plt.savefig(os.path.join(directory, f"plot_{i+1}.png"))                        
                elif isinstance(save_as, str):
                    plt.savefig(os.path.join(directory, f"{save_as}_{i+1}.png"))
                else:
                    plt.savefig(os.path.join(directory, f"plot_{i+1}.png"))
                plt.close(fig)
    else:
        if len(line_df.columns) <= 5:
            for i in range (0, len(line_df.columns)):
                # Plotting the Data     
                fig = plt.figure(figsize=fig_size, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111)                       
                plt.plot(time, line_df[line_df.columns[i]], linestyles[0], label=legend[0], linewidth = linewidth[0])
                plt.legend(fontsize=15)

                # Adjusting the plot if user wants tight x axis limits
                if padding:
                    plt.xlim(time[0], time[-1])

                plt.xticks(fontsize=15, rotation=45)
                plt.yticks(fontsize=15)

                # Placing Labels if requested
                if labels:
                    # Plotting Labels
                    plt.xlabel(labels[0], fontsize=18)
                    plt.ylabel(labels[1]+"m\u00B3/s", fontsize=18)
                if title:
                    title_dict = {'family': 'sans-serif',
                                'color': 'black',
                                'weight': 'normal',
                                'size': 20,
                                }
                    ## Check that the title is a list of strings or a single string
                    if isinstance(title, list):
                        try:
                            if title[i] == '':
                                    ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)
                            else:
                                ax.set_title(label=title[i], fontdict=title_dict, pad=25)
                        except IndexError:
                            ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)                        
                    elif isinstance(title, str):
                        ax.set_title(label=title, fontdict=title_dict, pad=25)

                # Placing a grid if requested
                if grid:
                    plt.grid(True)

                # Fixes issues with parts of plot being cut off
                plt.tight_layout()

                # save to file if requested 
                if save:
                    # Check if the directory exists
                    if not os.path.exists(directory):
                        # If the directory does not exist, create it
                        os.makedirs(directory)
                        print(f"Directory '{directory}' created.")
                    elif directory != ".":
                        print(f"Directory '{directory}' already exists.")
                    else:
                        print("Plots will be saved to current directory")
                    ## Check that the title is a list of strings or a single string
                    if isinstance(save_as, list):
                        try:
                            if save_as[i] == '':
                                plt.savefig(os.path.join(directory, f"plot_{i+1}.png"))
                            else:
                                plt.savefig(os.path.join(directory, f"{save_as[i]}.png"))
                        except IndexError:
                            plt.savefig(os.path.join(directory, f"plot_{i+1}.png"))                        
                    elif isinstance(save_as, str):
                        plt.savefig(os.path.join(directory, f"{save_as}_{i+1}.png"))
                    else:
                        plt.savefig(os.path.join(directory, f"plot_{i+1}.png"))
        else:
            for i in range (0, len(line_df.columns)):
                # Plotting the Data     
                fig = plt.figure(figsize=fig_size, facecolor = 'w', edgecolor='k')
                ax = fig.add_subplot(111)                       
                plt.plot(time, line_df[line_df.columns[i]], linestyles[0], label=legend[0], linewidth = linewidth[0])
                plt.legend(fontsize=15)

                # Adjusting the plot if user wants tight x axis limits
                if padding:
                    plt.xlim(time[0], time[-1])

                plt.xticks(fontsize=15, rotation=45)
                plt.yticks(fontsize=15)

                # Placing Labels if requested
                if labels:
                    # Plotting Labels
                    plt.xlabel(labels[0], fontsize=18)
                    plt.ylabel(labels[1]+"m\u00B3/s", fontsize=18)
                if title:
                    title_dict = {'family': 'sans-serif',
                                'color': 'black',
                                'weight': 'normal',
                                'size': 20,
                                }
                    ## Check that the title is a list of strings or a single string
                    if isinstance(title, list):
                        try:
                            if title[i] == '':
                                    ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)
                            else:
                                ax.set_title(label=title[i], fontdict=title_dict, pad=25)
                        except IndexError:
                            ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)                        
                    elif isinstance(title, str):
                        ax.set_title(label=title, fontdict=title_dict, pad=25)

                # Placing a grid if requested
                if grid:
                    plt.grid(True)

                # Fixes issues with parts of plot being cut off
                plt.tight_layout()
                
                # Check if the directory exists
                if not os.path.exists(directory):
                    # If the directory does not exist, create it
                    os.makedirs(directory)
                    print(f"Directory '{directory}' created.")
                elif directory != ".":
                    print(f"Directory '{directory}' already exists.")
                else:
                    print("Plots will be saved to current directory")
                ## Check that the title is a list of strings or a single string
                if isinstance(save_as, list):
                    try:
                        if save_as[i] == '':
                            plt.savefig(os.path.join(directory, f"plot_{i+1}.png"))
                        else:
                            plt.savefig(os.path.join(directory, f"{save_as[i]}.png"))
                    except IndexError:
                        plt.savefig(os.path.join(directory, f"plot_{i+1}.png"))                        
                elif isinstance(save_as, str):
                    plt.savefig(os.path.join(directory, f"{save_as}_{i+1}.png"))
                else:
                    plt.savefig(os.path.join(directory, f"plot_{i+1}.png"))
                plt.close(fig)




def bounded_plot(lines: Union[list[pd.DataFrame]], upper_bounds: list[pd.DataFrame] = None, lower_bounds: list[pd.DataFrame] = None,
         legend: tuple[str, str] = ('Simulated Data', 'Observed Data'), grid: bool = False, title: str = None,
         labels: tuple[str, str] = None, linestyles: tuple[str, str] = ('r-', 'b-'), padding: bool = False ,
         fig_size: tuple[float, float] = (10,6), transparency: tuple[float, float] = [0.4, 0.4], save:bool = False,
         save_as:str = None, directory:str = ".") ->plt.figure:
    """ Create a comparison time series line plot of simulated and observed time series data with optional
    upper and lower bounds 

    Parameters
    ----------
    lines : list[pd.DataFrame] or pd.DataFrame
        a single dataframe or a list containing the dataframes that will be plotted as lines
    
    upper_bounds : list[pd.DataFrame] or pd.DataFrame
        a single dataframe or a list containing the dataframes that will be plotted as upper bounds
        that should sync directly with the indexes on the lower bound dataframes

    lower_bounds : list[pd.DataFrame] or pd.DataFrame
        a single dataframe or a list containing the dataframes that will be plotted as lower bounds
        that should sync directly with the indexes on the upper bound dataframes

    legend: tuple[str, str]
        Adds a Legend in the 'best' location determined by matplotlib.

    grid: bool
        If True, adds a grid to the plot.

    title: str
        If given, adds a title to the plot.

    labels: tuple[str, str]
        List of two str type inputs specifying x-axis labels and y-axis labels, respectively.

    linestyles: tuple[str, str]
        List of two string type inputs thet will change the linestyle of the simulated and
        recorded data, respectively.

    padding: bool
        If true, will set the padding to zero for the lines in the line plot.

    fig_size: tuple[float, float]
        Tuple of length two that specifies the horizontal and vertical lengths of the plot in
        inches, respectively.
    
    transparency: tuple[float, float]
        a float between 0 and 0.9 that represents how bold the shaded bounds will be of the simulated
        and observed ranges respectively

    save: bool
        If True, the plot images will be saved as png files in the format bounded-plot_1.png, bounded-plot_2.png,
        etc., depending how many plots are generated or with the names specified in the save_as 
        variable
    
    save_as: str or list[str]
        If provided these wil be the naming nomenclature used to save the figures as specified by the
        save variable

    directory: str
        The directory where the plots will be saved. The current directory will be searched
        for the spedified 'directory' directory, if none is found, it will be created. Its default is the
        current directory represented by '.' .

    Returns
    -------
    fig : Matplotlib figure instance
    
    Examples
    --------
    Visualization of the bounded long term seasonal data of a station with
    its median bounedd by its max and min 
    """

    ## Check that the inputs are lists of DataFrames or DataFrames
    if isinstance(lines, pd.DataFrame):
        lines = [lines]
    elif not isinstance(lines, list):
        raise ValueError("Argument must be a dataframe or a list of dataframes.")
    
    if upper_bounds is None:
        upper_bounds = []
    elif isinstance(upper_bounds, pd.DataFrame):
        upper_bounds = [upper_bounds]
    elif not isinstance(upper_bounds, list):
        raise ValueError("Argument must be a dataframe or a list of dataframes.")
    
    if lower_bounds is None:
        lower_bounds = []
    elif isinstance(lower_bounds, pd.DataFrame):
        lower_bounds = [lower_bounds]
    elif not isinstance(lower_bounds, list):
        raise ValueError("Argument must be a dataframe or a list of dataframes.")

    if len(lower_bounds) != len(upper_bounds):
        raise ValueError("You must have the same number of upper and lower bounds")



    # Plotting
    for line in lines:
        if not isinstance(line, pd.DataFrame):
            raise ValueError("All items in the list must be a DataFrame.")
        
        # Setting Variable for the simulated data, observed data, and time stamps
        line_obs = line.iloc[:, [0]]
        line_sim = line.iloc[:, [1]]
        for k in range(2, len(line.columns), 2):
            line_obs = pd.concat([line_obs, line.iloc[:, k]], axis = 1)
            line_sim = pd.concat([line_sim, line.iloc[:, k+1]], axis = 1)
        time = line.index

        if len(line_obs.columns) <= 5:
            for i in range (0, len(line_obs.columns)):
                fig = plt.figure(figsize=fig_size, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111)        

                # Plotting the line data
                plt.plot(time, line_obs[line_obs.columns[i]], linestyles[1], label=legend[1], linewidth = 1.5)
                plt.plot(time, line_sim[line_sim.columns[i]], linestyles[0], label=legend[0], linewidth = 1.5)
                plt.legend(fontsize=15)
                        
                # check that there are bounds
                if upper_bounds or lower_bounds:
                    # lists to store all bounds
                    upper_obs = []
                    lower_obs = []
                    upper_sim = []
                    lower_sim = []

                    for j in range(0, len(upper_bounds)):
                        # check that the bounds are valid 
                        if not line.index.equals(upper_bounds[j].index) or not line.index.equals(lower_bounds[j].index):
                            raise ValueError("The lines and bounds don't have the same indexing syntax")
                        if not isinstance(upper_bounds[j], pd.DataFrame) and not isinstance(lower_bounds[j], pd.DataFrame):
                            raise ValueError("All items in the lists must be a DataFrame.")              
                    
                        # Setting Variable for the simulated data, observed data, and time stamps for the bounds
                        # Upper bounds
                        obs = upper_bounds[j].iloc[:, [0]]
                        sim = upper_bounds[j].iloc[:, [1]]
                        for k in range(2, len(upper_bounds[j].columns), 2):
                            obs = pd.concat([obs, upper_bounds[j].iloc[:, k]], axis = 1)
                            sim = pd.concat([sim, upper_bounds[j].iloc[:, k+1]], axis = 1)
                        upper_sim.append(sim)
                        upper_obs.append(obs)

                        # Lower bounds
                        obs = lower_bounds[j].iloc[:, [0]]
                        sim = lower_bounds[j].iloc[:, [1]]
                        for k in range(2, len(lower_bounds[j].columns), 2):
                            obs = pd.concat([obs, lower_bounds[j].iloc[:, k]], axis = 1)
                            sim = pd.concat([sim, lower_bounds[j].iloc[:, k+1]], axis = 1)
                        lower_sim.append(sim)
                        lower_obs.append(obs)

                    # Plotting the range data                
                    idx = len(upper_bounds)-1
                    while idx > 0:
                        # Observed Data
                        plt.fill_between(time, upper_obs[idx-1][upper_obs[idx-1].columns[i]], upper_obs[idx][upper_obs[idx].columns[i]],
                                        alpha=(transparency[1]/len(upper_bounds))*(len(upper_bounds)-idx), color=linestyles[1][0])
                        plt.fill_between(time, lower_obs[idx][lower_obs[idx].columns[i]], lower_obs[idx-1][lower_obs[idx-1].columns[i]],
                                        alpha=(transparency[1]/len(upper_bounds))*(len(upper_bounds)-idx), color=linestyles[1][0])
                        # Simulated Data
                        plt.fill_between(time, upper_sim[idx-1][upper_sim[idx-1].columns[i]], upper_sim[idx][upper_sim[idx].columns[i]],
                                        alpha=(transparency[0]/len(upper_bounds))*(len(upper_bounds)-idx), color=linestyles[0][0])
                        plt.fill_between(time, lower_sim[idx][lower_sim[idx].columns[i]], lower_sim[idx-1][lower_sim[idx-1].columns[i]],
                                        alpha=(transparency[0]/len(upper_bounds))*(len(upper_bounds)-idx), color=linestyles[0][0])
                        idx-=1
                    
                    # Observed Data
                    plt.fill_between(time, line_obs[line_obs.columns[i]], upper_obs[0][upper_obs[0].columns[i]], alpha=transparency[1], color=linestyles[1][0])
                    plt.fill_between(time, lower_obs[0][lower_obs[0].columns[i]], line_obs[line_obs.columns[i]], alpha=transparency[1], color=linestyles[1][0])
                    # Simulated Data
                    plt.fill_between(time, line_sim[line_sim.columns[i]], upper_sim[0][upper_sim[0].columns[i]], alpha=transparency[0], color=linestyles[0][0])
                    plt.fill_between(time, lower_sim[0][lower_sim[0].columns[i]], line_sim[line_sim.columns[i]], alpha=transparency[0], color=linestyles[0][0])


                    
                # Adjusting the plot if user wants tight x axis limits
                if padding:
                    plt.xlim(time[0], time[-1])

                plt.xticks(fontsize=15, rotation=45)
                plt.yticks(fontsize=15)

                # Placing Labels if requested
                if labels:
                    # Plotting Labels
                    plt.xlabel(labels[0], fontsize=18)
                    plt.ylabel(labels[1], fontsize=18)
                if title:
                    title_dict = {'family': 'sans-serif',
                                'color': 'black',
                                'weight': 'normal',
                                'size': 20,
                                }
                    ## Check that the title is a list of strings or a single string
                    if isinstance(title, list):
                        try:
                            if title[i] == '':
                                ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)
                            else:
                                ax.set_title(label=title[i], fontdict=title_dict, pad=25)
                        except IndexError:
                            ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)
                    elif isinstance(title, str):
                        ax.set_title(label=title, fontdict=title_dict, pad=25)

                # Placing a grid if requested
                if grid:
                    plt.grid(True)

                # Fixes issues with parts of plot being cut off
                plt.tight_layout()

                # save to file if requested 
                if save:
                    # fig.set_facecolor('gainsboro')
                    # Check if the directory exists
                    if not os.path.exists(directory):
                        # If the directory does not exist, create it
                        os.makedirs(directory)
                        print(f"Directory '{directory}' created.")
                    elif directory != ".":
                        print(f"Directory '{directory}' already exists.")
                    else:
                        print("Plots will be saved to current directory")
                    ## Check that the title is a list of strings or a single string
                    if isinstance(save_as, list):
                        try:
                            if save_as[i] == '':
                                plt.savefig(os.path.join(directory, f"bounded-plot_{i+1}.png"))
                            else:
                                plt.savefig(os.path.join(directory, f"{save_as[i]}.png"))
                        except IndexError:
                            plt.savefig(os.path.join(directory, f"bounded-plot_{i+1}.png"))                        
                    elif isinstance(save_as, str):
                        plt.savefig(os.path.join(directory, f"{save_as}_{i+1}.png"))
                    else:
                        plt.savefig(os.path.join(directory, f"bounded-plot_{i+1}.png"))
        else:
            for i in range (0, len(line_obs.columns)):
                fig = plt.figure(figsize=fig_size, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111)        

                # Plotting the line data
                plt.plot(time, line_obs[line_obs.columns[i]], linestyles[1], label=legend[1], linewidth = 1.5)
                plt.plot(time, line_sim[line_sim.columns[i]], linestyles[0], label=legend[0], linewidth = 1.25)
                plt.legend(fontsize=15)
                        
                # check that there are bounds
                if upper_bounds or lower_bounds:
                    # lists to store all bounds
                    upper_obs = []
                    upper_sim = []
                    lower_obs = []
                    lower_sim = []

                    for j in range(0, len(upper_bounds)):
                        # check that the bounds are valid 
                        if not line.index.equals(upper_bounds[j].index) or not line.index.equals(lower_bounds[j].index):
                            raise ValueError("The lines and bounds don't have the same indexing syntax")
                        if not isinstance(upper_bounds[j], pd.DataFrame) and not isinstance(lower_bounds[j], pd.DataFrame):
                            raise ValueError("All items in the lists must be a DataFrame.")              
                    
                        # Setting Variable for the simulated data, observed data, and time stamps for the bounds
                        # Upper bounds
                        obs = upper_bounds[j].iloc[:, [0]]
                        sim = upper_bounds[j].iloc[:, [1]]
                        for k in range(2, len(upper_bounds[j].columns), 2):
                            obs = pd.concat([obs, upper_bounds[j].iloc[:, k]], axis = 1)
                            sim = pd.concat([sim, upper_bounds[j].iloc[:, k+1]], axis = 1)
                        upper_sim.append(sim)
                        upper_obs.append(obs)

                        # Lower bounds
                        obs = lower_bounds[j].iloc[:, [0]]
                        sim = lower_bounds[j].iloc[:, [1]]
                        for k in range(2, len(lower_bounds[j].columns), 2):
                            obs = pd.concat([obs, lower_bounds[j].iloc[:, k]], axis = 1)
                            sim = pd.concat([sim, lower_bounds[j].iloc[:, k+1]], axis = 1)
                        lower_sim.append(sim)
                        lower_obs.append(obs)

                    # Plotting the range data                
                    idx = len(upper_bounds)-1
                    while idx > 0:
                        # Observed Data
                        plt.fill_between(time, upper_obs[idx-1][upper_obs[idx-1].columns[i]], upper_obs[idx][upper_obs[idx].columns[i]],
                                        alpha=(transparency[1]/len(upper_bounds))*(len(upper_bounds)-idx), color=linestyles[1][0])
                        plt.fill_between(time, lower_obs[idx][lower_obs[idx].columns[i]], lower_obs[idx-1][lower_obs[idx-1].columns[i]],
                                        alpha=(transparency[1]/len(upper_bounds))*(len(upper_bounds)-idx), color=linestyles[1][0])
                        # Simulated Data
                        plt.fill_between(time, upper_sim[idx-1][upper_sim[idx-1].columns[i]], upper_sim[idx][upper_sim[idx].columns[i]],
                                        alpha=(transparency[0]/len(upper_bounds))*(len(upper_bounds)-idx), color=linestyles[0][0])
                        plt.fill_between(time, lower_sim[idx][lower_sim[idx].columns[i]], lower_sim[idx-1][lower_sim[idx-1].columns[i]],
                                        alpha=(transparency[0]/len(upper_bounds))*(len(upper_bounds)-idx), color=linestyles[0][0])
                        idx-=1
                    
                    # Observed Data
                    plt.fill_between(time, line_obs[line_obs.columns[i]], upper_obs[0][upper_obs[0].columns[i]], alpha=transparency[1], color=linestyles[1][0])
                    plt.fill_between(time, lower_obs[0][lower_obs[0].columns[i]], line_obs[line_obs.columns[i]], alpha=transparency[1], color=linestyles[1][0])
                    # Simulated Data
                    plt.fill_between(time, line_sim[line_sim.columns[i]], upper_sim[0][upper_sim[0].columns[i]], alpha=transparency[0], color=linestyles[0][0])
                    plt.fill_between(time, lower_sim[0][lower_sim[0].columns[i]], line_sim[line_sim.columns[i]], alpha=transparency[0], color=linestyles[0][0])


                    
                # Adjusting the plot if user wants tight x axis limits
                if padding:
                    plt.xlim(time[0], time[-1])

                plt.xticks(fontsize=15, rotation=45)
                plt.yticks(fontsize=15)

                # Placing Labels if requested
                if labels:
                    # Plotting Labels
                    plt.xlabel(labels[0], fontsize=18)
                    plt.ylabel(labels[1], fontsize=18)
                if title:
                    title_dict = {'family': 'sans-serif',
                                'color': 'black',
                                'weight': 'normal',
                                'size': 20,
                                }
                    ## Check that the title is a list of strings or a single string
                    if isinstance(title, list):
                        try:
                            if title[i] == '':
                                ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)
                            else:
                                ax.set_title(label=title[i], fontdict=title_dict, pad=25)
                        except IndexError:
                            ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)                            
                    elif isinstance(title, str):
                        ax.set_title(label=title, fontdict=title_dict, pad=25)

                # Placing a grid if requested
                if grid:
                    plt.grid(True)

                # Fixes issues with parts of plot being cut off
                plt.tight_layout()

                # Check if the directory exists
                if not os.path.exists(directory):
                    # If the directory does not exist, create it
                    os.makedirs(directory)
                    print(f"Directory '{directory}' created.")
                elif directory != ".":
                    print(f"Directory '{directory}' already exists.")
                else:
                    print("Plots will be saved to current directory")
                ## Check that the title is a list of strings or a single string
                if isinstance(save_as, list):
                    try:
                        if save_as[i] == '':
                            plt.savefig(os.path.join(directory, f"bounded-plot_{i+1}.png"))
                        else:
                            plt.savefig(os.path.join(directory, f"{save_as[i]}.png"))
                    except IndexError:
                        plt.savefig(os.path.join(directory, f"bounded-plot_{i+1}.png"))                        
                elif isinstance(save_as, str):
                    plt.savefig(os.path.join(directory, f"{save_as}_{i+1}.png"))
                else:
                    plt.savefig(os.path.join(directory, f"bounded-plot_{i+1}.png"))
                plt.close(fig)
        

def histogram():
    return

def scatter(grid: bool = False, title: str = None, labels: tuple[str, str] = None,
         fig_size: tuple[float, float] = (10,6), best_fit: bool=False, line45: bool=False,

         merged_df: pd.DataFrame = None, obs_df: pd.DataFrame =  None, sim_df: pd.DataFrame = None,
         metrices: list[str] = None, markerstyle: str = 'ko', save: bool=False, plot_adjust: float = 0.2,
         save_as:str = None, metrics_adjust: tuple[float, float] = (1.05, 0.5), directory:str = ".",

         shapefile_path: str = "", x_axis : pd.DataFrame=None, y_axis : pd.DataFrame=None,
         metric: str="", observed: pd.DataFrame = None, simulated: pd.DataFrame = None)-> plt.figure:
    """ Creates a scatter plot of the observed and simulated data.

    Parameters
    ----------
    grid: bool
        If True, adds a grid to the plot.

    title: str
        If given, adds a title to the plot.

    labels: tuple[str, str]
        List of two string inputs specifying x-axis labels and y-axis labels, respectively.

    fig_size: tuple[float, float]
        Tuple of length two that specifies the horizontal and vertical lengths of the plot in
        inches, respectively.

    merged_df : pd.DataFrame
        the dataframe containing the series of observed and simulated values. It must have a datetime
        index and an even number of columns where in any two, the left column is the Measured/observed
        data and the right is the Simulated data. If it is present, the obs_df and sim_df must be None.
    
    obs_df : pd.DataFrame
        A DataFrame conataning rows of measured data. It must have a datetime index. if it is
        present it is accompanied by the sim_df and the merged_df must be None.

    sim_df : pd.DataFrame
        A DataFrame conataning rows of predicted/simulated data. It must have a datetime index. if it is
        present it is accompanied by the obs_df and the merged_df must be None.

    metrices: list[str]
        Adds Metrics to the left side of the plot. Any metric from the postprocessing.metrics library
        can be added to the plot as the abbreviation of the function. The entries must be in a list.
        (e.g. ['PBIAS', 'MSE', 'KGE']).

    markerstyle: str
        List of two strings that determine the point style and shape of the data being plotted 

    metrics_adjust: tuple[float, float]
        Tuple of length two with float inputs indicating the relative position of the text (x-coordinate,
        y-coordinate) when adding metrics to the plot.

    plot_adjust: float
        Specifies the relative position to shift the plot to the left when adding metrics to the
        plot. 

    best_fit: bool
        If True, adds a best linear regression line on the graph with the equation for the line in the legend. 

    line45: bool
        IF True, adds a 45 degree line to the plot and the legend. 
        
    save: bool
        If True, the plot images will be saved as png files in the format scatter-plot_1.png, scatter-plot_2.png,
        etc., depending how many plots are generated or with the names specified in the save_as 
        variable
    
    save_as: str or list[str]
        If provided these wil be the naming nomenclature used to save the figures as specified by the
        save variable 
    
    directory: str
        The directory where the plots will be saved. The current directory will be searched
        for the specified 'directory'. If none is found, it will be created. Its default is the
        current directory represented by '.' .

    shapefile_path : str
        Tha path to a shapefile on top of which you will be plotting the scatter plot

    x_axis: pd.DataFrame
        Used exclusively when plotting with a shapefile. It is used to determine the x-axis of the plot

    y_axis: pd.DataFrame
        Used exclusively when plotting with a shapefile. It is used to determine the y-axis of the plot

    metric: str
        This is the metric that is used to make and map the color map of the scatter plot

    observed: pd.DataFrame
        This is used to calculate the metric as stated above

    simulated: pd.DataFrame
        This is used to calculate the metric as shown above
    
    Returns
    -------
    fig : Matplotlib figure instance

    Examples
    --------
    Visualization of a station's data using a 2D plot

    >>> from postprocessinglib.evaluation import metrics, visuals, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> observed = DATAFRAMES["DF_OBSERVED"] 
    >>> simulated = DATAFRAMES["DF_SIMULATED"]
    >>> merged_df = DATAFRAMES["DF"]
    >>> .
    >>> # plot of the stations in the dataframe from 1981 - 1985
    >>> visuals.scatter(merged_df = merged_df['1981-01-01':'1985-12-31'],
               grid = True,
               labels = ("Simulated Data", "Observed Data"),
               markerstyle = 'b.',
               line45 = True,
               title = "Scatterplot of 1981 - 1985"
               metrices = ['KGE','MSE','BIAS']
               )

    .. image:: ../Figures/scatter_Station_1_1981_to_1990.png
    .. image:: ../Figures/scatter_Station_2_1981_to_1990.png

    >>> shapefile_path = r"SaskRB_SubDrainage2.shp"
    >>> stations_path = 'Station_data.xlsx'
    >>> Station_info = pd.read_excel(io=stations_path)
    >>> .
    >>> # plot of a few stations in the SRB showing the disparities in their KGE
    >>> visuals.scatter(shapefile_path = shapefile_path,
                        title = "SRB SubDrainage and KGE",
                        x_axis = Station_info["Lon"],
                        y_axis = Station_info["Lat"],
                        metric = "KGE",
                        fig_size = (24, 20),
                        observed = DATA_2["DF_OBSERVED"],
                        simulated = DATA_2["DF_SIMULATED"],
                        labels=['Longitude', 'Latitude'],
                    )

    .. image:: ../Figures/SRB_subDrainage_showing_KGE.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/Examples.ipynb>`_

    """     
    # Plotting the Data
    if not shapefile_path:
        if merged_df is not None and sim_df is None and obs_df is None:
            # Setting Variable for the simulated and observed data
            obs = merged_df.iloc[:, [0]]
            sim = merged_df.iloc[:, [1]]
            for j in range(2, len(merged_df.columns), 2):
                obs = pd.concat([obs, merged_df.iloc[:, j]], axis = 1)
                sim = pd.concat([sim, merged_df.iloc[:, j+1]], axis = 1)
        elif merged_df is None and obs_df is not None and sim_df is not None:
            obs = obs_df
            sim = sim_df
        else:
            raise RuntimeError('either sim_df and obs_df or merged_df are required inputs.')

        if len(obs.columns) <= 5:
            for i in range (0, len(obs.columns)):
                # Plotting the Data
                fig = plt.figure(figsize=fig_size, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111) 
                plt.plot(sim[sim.columns[i]], obs[obs.columns[i]], markerstyle)
                plt.xticks(fontsize=15, rotation=45)
                plt.yticks(fontsize=15)

                if best_fit:
                    # Getting a polynomial fit and defining a function with it
                    p = np.polyfit(sim.iloc[:, i], obs.iloc[:, i], 1)
                    f = np.poly1d(p)

                    # Calculating new x's and y's
                    x_new = np.linspace(sim.iloc[:, i].min(), sim.iloc[:, i].max(), sim.size)
                    y_new = f(x_new)

                    # Formatting the best fit equation to be able to display in latex
                    equation = "{} x + {}".format(np.round(p[0], 4), np.round(p[1], 4))

                    # Plotting the best fit line with the equation as a legend in latex
                    plt.plot(x_new, y_new, 'r', label="${}$".format(equation))

                
                if line45:
                    max = np.nanmax([sim.iloc[:, i].max(), obs.iloc[:, i].max()])
                    plt.plot(np.arange(0, int(max) + 1), np.arange(0, int(max) + 1), 'r--', label='45$^\u00b0$ Line')

                
                if best_fit or line45:
                    plt.legend(fontsize=12)
                
                # Placing Labels if requested
                if labels:
                    # Plotting Labels
                    plt.xlabel(labels[0], fontsize=12)
                    plt.ylabel(labels[1], fontsize=12)

                if title:
                    title_dict = {'family': 'sans-serif',
                                'color': 'black',
                                'weight': 'normal',
                                'size': 20,
                                }
                    ## Check that the title is a list of strings or a single string
                    if isinstance(title, list):
                        try:
                            if title[i] == '':
                                ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)
                            else:
                                ax.set_title(label=title[i], fontdict=title_dict, pad=25)
                        except IndexError:
                            ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)                            
                    elif isinstance(title, str):
                        ax.set_title(label=title, fontdict=title_dict, pad=25)

                # Placing a grid if requested
                if grid:
                    plt.grid(True)
                
                plt.tight_layout()

                # Placing Metrics on the Plot if requested
                if metrices:
                    formatted_selected_metrics = 'Metrics: \n'
                    if metrices == 'all':
                        for key, value in metrics.calculate_all_metrics(observed=obs, simulated=sim).items():
                            formatted_selected_metrics += key + ' : ' + str(value[i]) + '\n'
                    else: 
                        assert isinstance(metrices, list)
                        for key, value in metrics.calculate_metrics(observed=obs, simulated=sim, metrices=metrices).items():
                            formatted_selected_metrics += key + ' : ' + str(value[i]) + '\n'

                    font = {'family': 'sans-serif',
                            'weight': 'normal',
                            'size': 12}
                    plt.text(metrics_adjust[0], metrics_adjust[1], formatted_selected_metrics, ha='left',
                        va='center', transform=ax.transAxes, fontdict=font, #mouseover = True,
                        bbox = dict(boxstyle = "round4, pad = 0.6,rounding_size=0.3", facecolor = "0.8", edgecolor="k"))

                    plt.subplots_adjust(right = 1-plot_adjust)

                # save to file if requested 
                if save:
                    # fig.set_facecolor('gainsboro')
                    # Check if the directory exists
                    if not os.path.exists(directory):
                        # If the directory does not exist, create it
                        os.makedirs(directory)
                        print(f"Directory '{directory}' created.")
                    elif directory != ".":
                        print(f"Directory '{directory}' already exists.")
                    else:
                        print("Plots will be saved to current directory")
                    ## Check that the title is a list of strings or a single string
                    if isinstance(save_as, list):
                        try:
                            if save_as[i] == '':
                                plt.savefig(os.path.join(directory, f"scatter-plot_{i+1}.png"))
                            else:
                                plt.savefig(os.path.join(directory, f"{save_as[i]}.png"))
                        except IndexError:
                            plt.savefig(os.path.join(directory, f"scatter-plot_{i+1}.png"))                        
                    elif isinstance(save_as, str):
                        plt.savefig(os.path.join(directory, f"{save_as}_{i+1}.png"))
                    else:
                        plt.savefig(os.path.join(directory, f"scatter-plot_{i+1}.png"))
        else:
            for i in range (0, len(obs.columns)):
                # Plotting the Data
                fig = plt.figure(figsize=fig_size, facecolor="w", edgecolor='k')
                ax = fig.add_subplot(111) 
                plt.plot(sim[sim.columns[i]], obs[obs.columns[i]], markerstyle)
                plt.xticks(fontsize=15, rotation=45)
                plt.yticks(fontsize=15)

                if best_fit:
                    # Getting a polynomial fit and defining a function with it
                    p = np.polyfit(sim.iloc[:, i], obs.iloc[:, i], 1)
                    f = np.poly1d(p)

                    # Calculating new x's and y's
                    x_new = np.linspace(sim.iloc[:, i].min(), sim.iloc[:, i].max(), sim.size)
                    y_new = f(x_new)

                    # Formatting the best fit equation to be able to display in latex
                    equation = "{} x + {}".format(np.round(p[0], 4), np.round(p[1], 4))

                    # Plotting the best fit line with the equation as a legend in latex
                    plt.plot(x_new, y_new, 'r', label="${}$".format(equation))

                
                if line45:
                    max = np.nanmax([sim.iloc[:, i].max(), obs.iloc[:, i].max()])
                    plt.plot(np.arange(0, int(max) + 1), np.arange(0, int(max) + 1), 'r--', label='45$^\u00b0$ Line')

                
                if best_fit or line45:
                    plt.legend(fontsize=12)
                
                # Placing Labels if requested
                if labels:
                    # Plotting Labels
                    plt.xlabel(labels[0], fontsize=12)
                    plt.ylabel(labels[1], fontsize=12)

                if title:
                    title_dict = {'family': 'sans-serif',
                                'color': 'black',
                                'weight': 'normal',
                                'size': 20,
                                }
                    ## Check that the title is a list of strings or a single string
                    if isinstance(title, list):
                        try:
                            if title[i] == '':
                                ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)
                            else:
                                ax.set_title(label=title[i], fontdict=title_dict, pad=25)
                        except IndexError:
                            ax.set_title(label='Hydrograph of the time series', fontdict=title_dict, pad=25)                            
                    elif isinstance(title, str):
                        ax.set_title(label=title, fontdict=title_dict, pad=25)

                # Placing a grid if requested
                if grid:
                    plt.grid(True)
                
                plt.tight_layout()

                # Placing Metrics on the Plot if requested
                if metrices:
                    formatted_selected_metrics = 'Metrics: \n'
                    if metrices == 'all':
                        for key, value in metrics.calculate_all_metrics(observed=obs, simulated=sim).items():
                            formatted_selected_metrics += key + ' : ' + str(value[i]) + '\n'
                    else: 
                        assert isinstance(metrices, list)
                        for key, value in metrics.calculate_metrics(observed=obs, simulated=sim, metrices=metrices).items():
                            formatted_selected_metrics += key + ' : ' + str(value[i]) + '\n'

                    font = {'family': 'sans-serif',
                            'weight': 'normal',
                            'size': 12}
                    plt.text(metrics_adjust[0], metrics_adjust[1], formatted_selected_metrics, ha='left',
                        va='center', transform=ax.transAxes, fontdict=font, #mouseover = True,
                        bbox = dict(boxstyle = "round4, pad = 0.6,rounding_size=0.3", facecolor = "0.8", edgecolor="k"))

                    plt.subplots_adjust(right = 1-plot_adjust)
                
                # Check if the directory exists
                if not os.path.exists(directory):
                    # If the directory does not exist, create it
                    os.makedirs(directory)
                    print(f"Directory '{directory}' created.")
                elif directory != ".":
                    print(f"Directory '{directory}' already exists.")
                else:
                    print("Plots will be saved to current directory")
                ## Check that the title is a list of strings or a single string
                if isinstance(save_as, list):
                    try:
                        if save_as[i] == '':
                            plt.savefig(os.path.join(directory, f"scatter-plot_{i+1}.png"))
                        else:
                            plt.savefig(os.path.join(directory, f"{save_as[i]}.png"))
                    except IndexError:
                        plt.savefig(os.path.join(directory, f"scatter-plot_{i+1}.png"))                        
                elif isinstance(save_as, str):
                    plt.savefig(os.path.join(directory, f"{save_as}_{i+1}.png"))
                else:
                    plt.savefig(os.path.join(directory, f"scatter-plot_{i+1}.png"))
                plt.close(fig)
    else:
        metr = metrics.calculate_metrics(observed=observed, simulated=simulated, metrices=[metric])
        data = {
            'latitude': y_axis.values,
            'longitude': x_axis.values,
            list(metr)[0] : metr[list(metr)[0]]
        }
        dataframe = pd.DataFrame(data)
 
        # Convert the pandas DataFrame into a GeoDataFrame
        geometry = [Point(xy) for xy in zip(dataframe['longitude'], dataframe['latitude'])]
        gdf_points = gpd.GeoDataFrame(dataframe, geometry=geometry)
        
        # Read the shapefile using GeoPandas
        gdf_shapefile = gpd.read_file(shapefile_path)
        
        fig= plt.figure(figsize=fig_size, dpi =150, frameon=True)
        ax = fig.add_axes([0,0.,.4,.1])
        gdf_shapefile.plot(ax=ax, edgecolor='black', facecolor= "None", linewidth=0.5, legend=True)
        
        # Plot the points with color based on 'kge' column
        sc = gdf_points.plot(ax=ax, column=list(metr)[0], cmap='jet', legend=True, markersize=40, legend_kwds={'label': list(metr)[0]+" Value", "orientation": "vertical"})

        # Placing Labels if requested
        if labels:
            # Plotting Labels
            plt.xlabel(labels[0], fontsize=12)
            plt.ylabel(labels[1], fontsize=12)

        if title:
            title_dict = {'family': 'sans-serif',
                        'color': 'black',
                        'weight': 'normal',
                        'size': 20,
                        }
            ax.set_title(label=title, fontdict=title_dict, pad=25)

        # Placing a grid if requested
        if grid:
            plt.grid(True)


def qqplot(grid: bool = False, title: str = None, labels: tuple[str, str] = None,
         fig_size: tuple[float, float] = (10,6), best_fit: bool=False,
         merged_df: pd.DataFrame = None, obs_df: pd.DataFrame =  None, sim_df: pd.DataFrame = None,
         metrices: list[str] = None, markerstyle: str = 'ko', save: bool=False, plot_adjust: float = 0.2,
         save_as:str = None, metrics_adjust: tuple[float, float] = (1.05, 0.5), directory:str = ".")-> plt.figure:
    """Plots a Quantile-Quantile plot of the simulated and observed data.

    Parameters
    ----------
    grid: bool
        If True, adds a grid to the plot.

    title: str
        If given, adds a title to the plot.

    labels: tuple[str, str]
        List of two string inputs specifying x-axis labels and y-axis labels, respectively.

    fig_size: tuple[float, float]
        Tuple of length two that specifies the horizontal and vertical lengths of the plot in
        inches, respectively.

    merged_df : pd.DataFrame
        the dataframe containing the series of observed and simulated values. It must have a datetime
        index and an even number of columns where in any two, the left column is the Measured/observed
        data and the right is the Simulated data. If it is present, the obs_df and sim_df must be None.
    
    obs_df : pd.DataFrame
        A DataFrame conataning rows of measured data. It must have a datetime index. if it is
        present it is accompanied by the sim_df and the merged_df must be None.

    sim_df : pd.DataFrame
        A DataFrame conataning rows of predicted/simulated data. It must have a datetime index. if it is
        present it is accompanied by the obs_df and the merged_df must be None.

    metrices: list[str]
        Adds Metrics to the left side of the plot. Any metric from the postprocessing.metrics library
        can be added to the plot as the abbreviation of the function. The entries must be in a list.
        (e.g. ['PBIAS', 'MSE', 'KGE']).

    markerstyle: str
        List of two strings that determine the point style and shape of the data being plotted 

    metrics_adjust: tuple[float, float]
        Tuple of length two with float inputs indicating the relative position of the text (x-coordinate,
        y-coordinate) when adding metrics to the plot.

    plot_adjust: float
        Specifies the relative position to shift the plot to the left when adding metrics to the
        plot. 

    best_fit: bool
        If True, adds a best linear regression line on the graph with the equation for the line in the legend. 

    line45: bool
        IF True, adds a 45 degree line to the plot and the legend. 
        
    save: bool
        If True, the plot images will be saved as png files in the format scatter-plot_1.png, scatter-plot_2.png,
        etc., depending how many plots are generated or with the names specified in the save_as 
        variable
    
    save_as: str or list[str]
        If provided these wil be the naming nomenclature used to save the figures as specified by the
        save variable 
    
    directory: str
        The directory where the plots will be saved. The current directory will be searched
        for the spedified 'directory' directory, if none is found, it will be created. Its default is the
        current directory represented by '.'

    Examples
    --------
    Visualization of a station's data using a 2D plot

    >>> from postprocessinglib.evaluation import metrics, visuals, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> observed = DATAFRAMES["DF_OBSERVED"] 
    >>> simulated = DATAFRAMES["DF_SIMULATED"]
    >>> merged_df = DATAFRAMES["DF"]
    >>> .
    >>> # plot of the stations in the dataframe from 1981 - 1985

    """

    if merged_df is not None and sim_df is None and obs_df is None:
        # Setting Variable for the simulated and observed data
        obs = merged_df.iloc[:, [0]].values
        sim = merged_df.iloc[:, [1]].values
        for j in range(2, len(merged_df.columns), 2):
            obs = pd.concat([obs, merged_df.iloc[:, j]], axis = 1)
            sim = pd.concat([sim, merged_df.iloc[:, j+1]], axis = 1)
    elif merged_df is None and obs_df is not None and sim_df is not None:
        obs = obs_df.values
        sim = sim_df.values
    else:
        raise RuntimeError('either sim_df and obs_df or merged_df are required inputs.')
    
    print(sim)
    print("\n centre line \n")
    print(obs)
    return
