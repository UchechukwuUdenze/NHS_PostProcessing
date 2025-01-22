"""
The visual module contains different plotting functions for time series visualization.
It allows users to plot hydrographs per station for each stations to allow us visualize
the time-series data

Some of them also allow their metrics to be placed beside the plots

"""

from typing import Union, List, Tuple
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from shapely.geometry import Point

from postprocessinglib.evaluation import metrics
from postprocessinglib.utilities import helper_functions as hlp

def _save_or_display_plot(fig, save: bool, save_as: Union[str, List[str]], dir: str, i: int, type: str):
    """Save the plot to a file or display it based on user preferences."""
    if save:
        plt.tight_layout()
        if not os.path.exists(dir):
            os.makedirs(dir)
        filename = f"{save_as}.png" if isinstance(save_as, str) else f"{type}_{i + 1}.png"
        fig.savefig(os.path.join(dir, filename))
        plt.close(fig)
    else:
        plt.show()

def _prepare_bounds(bounds: List[pd.DataFrame], col_index: int, observed: bool) -> List[pd.Series]:
    """
    Extracts the required column (observed or simulated) for all bounds.
    
    Args:
        bounds (List[pd.DataFrame]): List of bound DataFrames.
        col_index (int): Index of the column to extract.
        observed (bool): Whether to extract the observed (True) or simulated (False) columns.
    
    Returns:
        List[pd.Series]: Extracted columns from the bounds.
    """
    col_offset = 0 if observed else 1
    return [b.iloc[:, col_index * 2 + col_offset] for b in bounds]

def plot(
    merged_df: pd.DataFrame = None, 
    df: pd.DataFrame = None, 
    obs_df: pd.DataFrame = None, 
    sim_df: pd.DataFrame = None,
    legend: tuple[str, str] = ('Simulated Data', 'Observed Data'), 
    metrices: list[str] = None,
    grid: bool = False, 
    title: str = None, 
    labels: tuple[str, str] = None, 
    padding: bool = False,
    linestyles: tuple[str, str] = ('r-', 'b-'), 
    linewidth: tuple[float, float] = (1.5, 1.25),
    fig_size: tuple[float, float] = (10, 6), 
    metrics_adjust: tuple[float, float] = (1.05, 0.5),
    plot_adjust: float = 0.2, 
    save: bool = False, 
    save_as: str = None, 
    dir: str = os.getcwd()
    ) -> plt.figure:
    """ Create a comparison time series line plot of simulated and observed time series data

    Parameters
    ----------
    merged_df : pd.DataFrame, optional
        The dataframe containing the series of observed and simulated values. It must have a datetime index.
        
    obs_df : pd.DataFrame, optional
        A DataFrame containing the observed data series if using separate observed and simulated data.

    sim_df : pd.DataFrame, optional
        A DataFrame containing the simulated data series if using separate observed and simulated data.

    df : pd.DataFrame, optional
        A DataFrame containing the data to be plotted if no merged or separate observed/simulated data are provided.

    legend : tuple of str, optional
        A tuple containing the labels for the simulated and observed data, default is ('Simulated Data', 'Observed Data').

    metrices : list of str, optional
        A list of metrics to display on the plot, default is None.

    grid : bool, optional
        Whether to display a grid on the plot, default is False.

    title : str, optional
        The title of the plot.

    labels : tuple of str, optional
        A tuple containing the labels for the x and y axes.

    padding : bool, optional
        Whether to add padding to the x-axis limits for a tighter plot, default is False.

    linestyles : tuple of str, optional
        A tuple specifying the line styles for the simulated and observed data.

    linewidth : tuple of float, optional
        A tuple specifying the line widths for the simulated and observed data.

    fig_size : tuple of float, optional
        A tuple specifying the size of the figure.

    metrics_adjust : tuple of float, optional
        A tuple specifying the position for the metrics on the plot.

    plot_adjust : float, optional
        A value to adjust the plot layout to avoid clipping.

    save : bool, optional
        Whether to save the plot to a file, default is False.

    save_as : str or list of str, optional
        The name or list of names to save the plot as. If a list is provided, each plot will be saved with the corresponding name.

    dir : str, optional
        The directory to save the plot to, default is the current working directory.

    Returns
    -------
    fig : Matplotlib figure instance and/or png files of the figures.
    
    Examples
    --------

    >>> from postprocessinglib.evaluation import visuals
    >>> # Example 1: Plotting merged data with simulated and observed values
    >>> merged_data = pd.DataFrame({...})  # Your merged dataframe
    >>> visuals.plot(merged_df = merged_data,
                    title='Simulated vs Observed',
                    labels=['Time', 'Value'], grid=True,
                    metrices = ['KGE','RMSE'])

    .. image:: ../Figures/plot1_example.png

    >>> # Example 2: Plotting only observed and simulated data with custom linestyles and saving the plot
    >>> obs_data = pd.DataFrame({...})  # Your observed data
    >>> sim_data = pd.DataFrame({...})  # Your simulated data
    >>> visuals.plot(obs_df = obs_data, sim_df = sim_data, linestyles=('g-', 'b-'),
                    save=True, save_as="plot2_example", dir="../Figures")

    .. image:: ../Figures/plot2_example.png

    >>> # Example 3: Plotting a single dataframe
    >>> single_data = pd.DataFrame({...})  # Your single dataframe (either simulated or observed)
    >>> visuals.plot(df=single_data, grid=True, title="Single Line Plot", labels=("Time", "Value"))

    .. image:: ../Figures/plot3_example.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/Examples.ipynb>`_
         
    """
     # Assign the data based on inputs
    if merged_df is not None:
        # If merged_df is provided, separate observed and simulated data
        obs = merged_df.iloc[:, ::2]
        sim = merged_df.iloc[:, 1::2]
        time = merged_df.index
    elif sim_df is not None and obs_df is not None:
        # If both sim_df and obs_df are provided
        obs = obs_df
        sim = sim_df
        time = obs_df.index
    elif df is not None:
        # If only df is provided, treat it as both observed and simulated data
        obs = df # to keep the future for loop valid
        line_df = df
        time = df.index
    else:
        raise RuntimeError('Please provide valid data (merged_df, obs_df, sim_df, or df)')

    # Convert time index to float or int if not datetime
    if not isinstance(time, pd.DatetimeIndex):
        # if the index is not a datetime, then it was converted during aggregation to
        # either a string (most likely) or an int or float
        if (isinstance(time[0], int)) or (isinstance(time[0], float)):
            pass
        else:
            if '/' in time[0]:
                # daily
                time = [pd.Timestamp(datetime.datetime.strptime(day, '%Y/%j').date()) for day in time]
            elif '.' in time[0]:
                print(True)
                # weekly
                # datetime ignores the week specifier unless theres a weekday attached,
                # so we attach Sunday - day 0
                time = [week+'.0' for week in time]
                time = [pd.Timestamp(datetime.datetime.strptime(week, '%Y.%U.%w').date()) for week in time]
            elif '-' in time[0]:
                # monthly
                time = [pd.Timestamp(datetime.datetime.strptime(month, '%Y-%m').date()) for month in time]
            else: # yearly
                time = np.asarray(time, dtype='float')
    

    for i in range (0, len(obs.columns)):
        # Plotting the Data     
        fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')
        if df is not None:
            ax.plot(time, line_df.iloc[:, i], linestyles[0], label=legend[0], linewidth = linewidth[0])
        else:                                   
            ax.plot(time, obs.iloc[:, i], linestyles[1], label=legend[1], linewidth = linewidth[1])
            ax.plot(time, sim.iloc[:, i], linestyles[0], label=legend[0], linewidth = linewidth[0])
        ax.legend(fontsize=15)

        # Adjusting the plot limit and layout
        if padding:
            plt.xlim(time[0], time[-1])

        # Placing Labels, title and grid if requested
        plt.xticks(fontsize=10,
                   rotation=45)
        plt.yticks(fontsize=15)

        if labels:
            # Plotting Labels
            plt.xlabel(labels[0], fontsize=18)
            plt.ylabel(labels[1]+" (m\u00B3/s)", fontsize=18)
        
        if title:
            title_dict = {'family': 'sans-serif',
                        'color': 'black',
                        'weight': 'normal',
                        'size': 20,
                        }
            ## Check that the title is a list of strings or a single string
            if isinstance(title, list):
                ax.set_title(title[i] if i < len(title) else f"Plot {i + 1}", fontdict=title_dict, pad=25)
            elif isinstance(title, str):
                ax.set_title(label=title, fontdict=title_dict, pad=25)

        # Placing a grid if requested
        if grid:
            plt.grid(True)

        # Placing Metrics on the Plot if requested
        if metrices:
            formatted_selected_metrics = 'Metrics:\n'
            if df is None:
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

        # Save or auto-save for large column counts
        auto_save = len(obs.columns) > 5
        _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "plot")


def bounded_plot(
    lines: Union[List[pd.DataFrame], pd.DataFrame],
    upper_bounds: List[pd.DataFrame] = None,
    lower_bounds: List[pd.DataFrame] = None,
    legend: Tuple[str, str] = ('Simulated Data', 'Observed Data'),
    grid: bool = False,
    title: Union[str, List[str]] = None,
    labels: Tuple[str, str] = None,
    linestyles: Tuple[str, str] = ('r-', 'b-'),
    padding: bool = False,
    fig_size: Tuple[float, float] = (10, 6),
    transparency: Tuple[float, float] = (0.4, 0.4),
    save: bool = False,
    save_as: Union[str, List[str]] = None,
    dir: str = os.getcwd()
    ) -> plt.figure:
    """ 
    Plots time-series data with optional confidence bounds.

    This function generates line plots for observed and simulated data, along with shaded confidence bounds. 
    It supports flexible customization options, such as labels, legends, line styles, gridlines, and more. 
    If the number of columns exceeds a threshold, the plots are automatically saved instead of being displayed.

    Parameters
    ----------
    lines : list of pd.DataFrame
        A list of DataFrames containing the observed and simulated data series to be plotted. Each DataFrame must have a datetime index.

    upper_bounds : list of pd.DataFrame, optional
        A list of DataFrames containing the upper bounds for each series. If not provided, no upper bounds are plotted.

    lower_bounds : list of pd.DataFrame, optional
        A list of DataFrames containing the lower bounds for each series. If not provided, no lower bounds are plotted.

    legend : tuple of str, optional
        A tuple containing the labels for the simulated and observed data, default is ('Simulated Data', 'Observed Data').

    grid : bool, optional
        Whether to display a grid on the plot, default is False.

    title : str, optional
        The title of the plot.

    labels : tuple of str, optional
        A tuple containing the labels for the x and y axes.

    linestyles : tuple of str, optional
        A tuple specifying the line styles for the simulated and observed data.

    padding : bool, optional
        Whether to add padding to the x-axis limits for a tighter plot, default is False.

    fig_size : tuple of float, optional
        A tuple specifying the size of the figure.

    transparency : list of float, optional
        A list specifying the transparency levels for the upper and lower bounds, default is [0.4, 0.4].

    save : bool, optional
        Whether to save the plot to a file, default is False.

    save_as : str or list of str, optional
        The name or list of names to save the plot as. If a list is provided, each plot will be saved with the corresponding name.

    dir : str, optional
        The directory to save the plot to, default is the current working directory.

    Returns
    -------
    fig : Matplotlib figure instance
    
    Example
    -------
    Generate a bounded plot with simulated and observed data, along with upper and lower bounds.

    >>> import pandas as pd
    >>> import numpy as np
    >>> from postprocessinglib.evaluation import visuals

    # Create an index for the data
    >>> time_index = pd.date_range(start='2025-01-01', periods=50, freq='D')

    # Generate sample observed and simulated data
    >>> obs_data = pd.DataFrame({
    ...     "Station1_Observed": np.random.rand(50),
    ...     "Station2_Observed": np.random.rand(50)
    ... }, index=time_index)

    >>> sim_data = pd.DataFrame({
    ...     "Station1_Simulated": np.random.rand(50),
    ...     "Station2_Simulated": np.random.rand(50)
    ... }, index=time_index)

    # Combine observed and simulated data
    >>> data = pd.concat([obs_data, sim_data], axis=1)

    # Generate sample bounds
    >>> upper_bounds = [
    ...     pd.DataFrame({
    ...         "Station1_Upper": np.random.rand(50) + 0.5,
    ...         "Station2_Upper": np.random.rand(50) + 0.5
    ...     }, index=time_index)
    ... ]

    >>> lower_bounds = [
    ...     pd.DataFrame({
    ...         "Station1_Lower": np.random.rand(50) - 0.5,
    ...         "Station2_Lower": np.random.rand(50) - 0.5
    ...     }, index=time_index)
    ... ]

    # Plot the data with bounds
    >>> visuals.bounded_plot(
    ...     lines=data,
    ...     upper_bounds=upper_bounds,
    ...     lower_bounds=lower_bounds,
    ...     legend=('Simulated Data', 'Observed Data'),
    ...     labels=('Datetime', 'Streamflow'),
    ...     transparency = [0.4, 0.3],
    ...     grid=True,
    ...     save=True,
    ...     save_as = 'bounded_plot_example',
    ...     dir = '../Figures'
    ... )

    .. image:: ../Figures/bounded_plot_example_1.png

    # Adjust a few other metrics
    >>> visuals.bounded_plot(
    ...     lines = merged_df,
    ...     upper_bounds = upper_bounds,
    ...     lower_bounds = lower_bounds,
    ...     title=['Long Term Aggregation by days of the Year'],
    ...     legend = ['Predicted Streamflow','Recorded Streamflow'],
    ...     linestyles=['k', 'r-'],
    ...     labels=['Days of the year', 'Streamflow Values'],
    ...     transparency = [0.4, 0.7],
    ... )

    .. image:: ../Figures/bounded_plot_example_2.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/Examples.ipynb>`_
    
    """

    ## Check that the inputs are DataFrames
    if isinstance(lines, pd.DataFrame):
        lines = [lines]
    elif not isinstance(lines, list):
        raise ValueError("Argument must be a dataframe or a list of dataframes.")
    
    upper_bounds = upper_bounds or []
    lower_bounds = lower_bounds or []

    if not isinstance(upper_bounds, list) or not isinstance(lower_bounds, list):
        raise ValueError("Bounds must be lists of DataFrames.")
    if len(upper_bounds) != len(lower_bounds):
        raise ValueError("Upper and lower bounds lists must have the same length.")

    # Plotting
    for line in lines:
        if not isinstance(line, pd.DataFrame):
            raise ValueError("All items in the 'lines' must be a DataFrame.")
        
        # Setting Variable for the simulated data, observed data, and time stamps
        line_obs = line.iloc[:, ::2]
        line_sim = line.iloc[:, 1::2]
        time = line.index

        for i in range (0, len(line_obs.columns)):
            fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')
            ax.plot(time, line_obs.iloc[:, i], linestyles[1], label=legend[1], linewidth = 1.5)
            ax.plot(time, line_sim.iloc[:, i], linestyles[0], label=legend[0], linewidth = 1.5)
            ax.legend(fontsize=15)

            # Prepare bounds for the current column
            upper_obs = _prepare_bounds(upper_bounds, i, observed=True)
            lower_obs = _prepare_bounds(lower_bounds, i, observed=True)
            upper_sim = _prepare_bounds(upper_bounds, i, observed=False)
            lower_sim = _prepare_bounds(lower_bounds, i, observed=False)

            # Plot bounds
            for j in range(len(upper_bounds)):
                ax.fill_between(time, lower_obs[j], upper_obs[j], alpha=transparency[1], color=linestyles[1][0])
                ax.fill_between(time, lower_sim[j], upper_sim[j], alpha=transparency[0], color=linestyles[0][0])
            
            # Adjusting the plot limit and layout
            if padding:
                plt.xlim(time[0], time[-1])

            # Placing Labels, title and grid if requested
            plt.xticks(fontsize=15, rotation=45)
            plt.yticks(fontsize=15)

            if labels:
                # Plotting Labels
                plt.xlabel(labels[0], fontsize=18)
                plt.ylabel(labels[1]+" (m\u00B3/s)", fontsize=18)
            
            if title:
                title_dict = {'family': 'sans-serif',
                            'color': 'black',
                            'weight': 'normal',
                            'size': 20,
                            }
                ## Check that the title is a list of strings or a single string
                if isinstance(title, list):
                    ax.set_title(title[i] if i < len(title) else f"Bounded Plot {i + 1}", fontdict=title_dict, pad=25)
                elif isinstance(title, str):
                    ax.set_title(label=title, fontdict=title_dict, pad=25)

            # Placing a grid if requested
            if grid:
                plt.grid(True)

            # Save or auto-save for large column counts
            auto_save = len(line_obs.columns) > 5
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "bounded-plot")        

def histogram():
    return

def scatter(
  grid: bool = False, 
  title: str = None, 
  labels: tuple[str, str] = None,
  fig_size: tuple[float, float] = (10, 6), 
  best_fit: bool = False, 
  line45: bool = False,

  merged_df: pd.DataFrame = None, 
  obs_df: pd.DataFrame = None, 
  sim_df: pd.DataFrame = None,
  metrices: list[str] = None, 
  markerstyle: str = 'ko', 
  save: bool = False, 
  plot_adjust: float = 0.2,
  save_as: str = None, 
  metrics_adjust: tuple[float, float] = (1.05, 0.5), 
  dir: str = os.getcwd(),

  shapefile_path: str = "", 
  x_axis: pd.DataFrame = None, 
  y_axis: pd.DataFrame = None,
  metric: str = "", 
  observed: pd.DataFrame = None, 
  simulated: pd.DataFrame = None
  ) -> plt.figure:
    """ Creates a scatter plot of the observed and simulated data.

    Parameters
    ----------
    grid : bool, optional
        Whether to display a grid on the plot, default is False.

    title : str, optional
        The title of the plot.

    labels : tuple of str, optional
        A tuple containing the labels for the x and y axes.

    fig_size : tuple of float, optional
        A tuple specifying the size of the figure.

    merged_df : pd.DataFrame, optional
        The dataframe containing the series of observed and simulated values. It must have a datetime index.
        
    obs_df : pd.DataFrame, optional
        A DataFrame containing the observed data series if using separate observed and simulated data.

    sim_df : pd.DataFrame, optional
        A DataFrame containing the simulated data series if using separate observed and simulated data.

    metrices : list of str, optional
        A list of metrics to display on the plot, default is None.

    markerstyle: str
        List of two strings that determine the point style and shape of the data being plotted 

    metrics_adjust : tuple of float, optional
        A tuple specifying the position for the metrics on the plot.

    plot_adjust : float, optional
        A value to adjust the plot layout to avoid clipping. 

    best_fit: bool
        If True, adds a best linear regression line on the graph with the equation for the line in the legend. 

    line45: bool
        IF True, adds a 45 degree line to the plot and the legend. 
        
    save : bool, optional
        Whether to save the plot to a file, default is False.

    save_as : str or list of str, optional
        The name or list of names to save the plot as. If a list is provided, each plot will be saved with the corresponding name.

    dir : str, optional
        The directory to save the plot to, default is the current working directory.

    shapefile_path : str, optional
        The path to a shapefile on top of which the scatter plot will be drawn.

    x_axis : pd.DataFrame, optional
        Used when plotting with a shapefile to determine the x-axis values.

    y_axis : pd.DataFrame, optional
        Used when plotting with a shapefile to determine the y-axis values.

    metric : str, optional
        The metric used to generate the color map for the scatter plot.

    observed : pd.DataFrame, optional
        Used to calculate the metric for the scatter plot.

    simulated : pd.DataFrame, optional
        Used to calculate the metric for the scatter plot.
    
    Returns
    -------
    fig : Matplotlib figure instance

    Example
    -------
    Generate a scatter plot using observed and simulated data:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import visuals
    >>> #
    >>> # Create test data
    >>> index = pd.date_range(start="2022-01-01", periods=10, freq="D")
    >>> obs_df = pd.DataFrame({
    >>>     "Station1": np.random.rand(10),
    >>>     "Station2": np.random.rand(10)
    >>> }, index=index)
    >>> #
    >>> sim_df = pd.DataFrame({
    >>>     "Station1": np.random.rand(10),
    >>>     "Station2": np.random.rand(10)
    >>> }, index=index)
    >>> #
    >>> # Call the scatter plot function
    >>> metrics.scatter(
    >>>     obs_df=obs_df,
    >>>     sim_df=sim_df,
    >>>     labels=("Observed", "Simulated"),
    >>>     title="Scatter Plot Example",
    >>>     best_fit=True,
    >>>     line45=True,
    >>>     save=True,
    >>>     save_as="scatter_plot_example.png"
    >>> )

    .. image:: ../Figures/scatter_plot_example.png

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
        if merged_df is not None:
            # If merged_df is provided, separate observed and simulated data
            obs = merged_df.iloc[:, ::2]
            sim = merged_df.iloc[:, 1::2]
        elif sim_df is not None and obs_df is not None:
            # If both sim_df and obs_df are provided
            obs = obs_df
            sim = sim_df
        else:
            raise RuntimeError('Please provide valid data (merged_df, obs_df or sim_df)')

        for i in range (0, len(obs.columns)):
            # Plotting the Data
            fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k') 
            plt.plot(sim.iloc[:, i], obs.iloc[:, i], markerstyle)

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
            
            # Placing Labels, title and grid if requested
            plt.xticks(fontsize=15, rotation=45)
            plt.yticks(fontsize=15)

            # Placing a grid if requested
            if grid:
                plt.grid(True)

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
                    ax.set_title(title[i] if i < len(title) else f"Scatter Plot {i + 1}", fontdict=title_dict, pad=25)
                elif isinstance(title, str):
                    ax.set_title(label=title, fontdict=title_dict, pad=25)               

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

            # Save or auto-save for large column counts
            auto_save = len(obs.columns) > 5
        _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "scatter-plot")
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


def qqplot(
    grid: bool = False, 
    title: str = None, 
    labels: tuple[str, str] = None, 
    fig_size: tuple[float, float] = (10, 6),
    interpolate: str = "linear", 
    legend: bool = False, 
    linewidth: tuple[float, float] = (1, 2),
    merged_df: pd.DataFrame = None, 
    obs_df: pd.DataFrame = None, 
    sim_df: pd.DataFrame = None,
    linestyle: tuple[str, str, str] = ['bo', 'r-.', 'r-'], 
    quantile: tuple[int, int] = [25, 75],
    q_labels: tuple[str, str, str] = ['Quantiles', 'Range of Quantiles', 'Inter Quartile Range'],
    save: bool = False, 
    save_as: str = None, 
    dir: str = os.getcwd()
    ) -> plt.figure:
    """Plots a Quantile-Quantile plot of the simulated and observed data.

    Parameters
    ----------
    grid : bool, optional
        Whether to display a grid on the plot, default is False.

    title : str, optional
        The title of the plot.

    labels : tuple of str, optional
        A tuple containing the labels for the x and y axes

    fig_size: tuple[float, float]
        Tuple of length two that specifies the horizontal and vertical lengths of the plot in
        inches, respectively.

    interpolate: str
        Determines whether the quantiles should be interpolated when the data length differs.
        If True, the quantiles are interpolated to align the data lengths between the observed
        and simulated data, ensuring accurate comparison.
        Default is False.

    legend: bool
        Whether to display the legend or not. Default is False

    llinewidth : tuple of float, optional
        A tuple specifying the line widths for the simulated and observed data.

    merged_df : pd.DataFrame, optional
        The dataframe containing the series of observed and simulated values. It must have a datetime index.
        
    obs_df : pd.DataFrame, optional
        A DataFrame containing the observed data series if using separate observed and simulated data.

    sim_df : pd.DataFrame, optional
        A DataFrame containing the simulated data series if using separate observed and simulated data.

    linestyle: tuple[str, str, str]
        List of three strings that determine the point style and shape of the data being plotted 

    quantile: tuple[int, int]
        Range of quantiles to plot, with values between 0 and 1. The first value is the lower quantile,
        and the second is the upper. Default is (25, 75).
    
    q_labels: tuple[str, str, str]

    save : bool, optional
        Whether to save the plot to a file, default is False.

    save_as : str or list of str, optional
        The name or list of names to save the plot as. If a list is provided, each plot will be saved with the corresponding name.

    dir : str, optional
        The directory to save the plot to, default is the current working directory.

    Example
    -------
    Generate a QQ plot to compare observed and simulated data distributions:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> #
    >>> # Create test data
    >>> index = pd.date_range(start="2022-01-01", periods=10, freq="D")
    >>> obs_df = pd.DataFrame({
    >>>     "Station1": np.random.rand(10),
    >>>     "Station2": np.random.rand(10)
    >>> }, index=index)
    >>> #
    >>> sim_df = pd.DataFrame({
    >>>     "Station1": np.random.rand(10),
    >>>     "Station2": np.random.rand(10)
    >>> }, index=index)
    >>> #
    >>> # Call the QQ plot function
    >>> metrics.qqplot(
    >>>     obs_df=obs_df,
    >>>     sim_df=sim_df,
    >>>     labels=("Quantiles (Simulated)", "Quantiles (Observed)"),
    >>>     title="QQ Plot Example",
    >>>     save=True,
    >>>     save_as="qqplot_example.png"
    >>> )

    .. image:: ../Figures/qqplot_example.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/Examples.ipynb>`_

    """

    if merged_df is not None:
        # If merged_df is provided, separate observed and simulated data
        obs = merged_df.iloc[:, ::2]
        sim = merged_df.iloc[:, 1::2]
    elif sim_df is not None and obs_df is not None:
        # If both sim_df and obs_df are provided
        obs = obs_df
        sim = sim_df
    else:
        raise RuntimeError('Please provide valid data (merged_df, obs_df or sim_df)')

    for i in range (0, len(obs.columns)):
        n = obs.iloc[:, i].size
        pvec = 100 * ((np.arange(1, n + 1) - 0.5) / n)
        sim_perc = np.percentile(sim.iloc[:, i], pvec, method=interpolate)
        obs_perc = np.percentile(obs.iloc[:, i], pvec, method=interpolate)

        # Finding the interquartile range to plot the best fit line
        quant_1_sim = np.percentile(obs.iloc[:, i], quantile[0], interpolation=interpolate)
        quant_3_sim = np.percentile(obs.iloc[:, i], quantile[1], interpolation=interpolate)
        quant_1_obs = np.percentile(obs.iloc[:, i], quantile[0], interpolation=interpolate)
        quant_3_obs = np.percentile(obs.iloc[:, i], quantile[1], interpolation=interpolate)

        dsim = quant_3_sim - quant_1_sim
        dobs = quant_3_obs - quant_1_obs
        slope = dobs / dsim

        centersim = (quant_1_sim + quant_3_sim) / 2
        centerobs = (quant_1_obs + quant_3_obs) / 2
        maxsim = np.max(obs.iloc[:, i])
        minsim = np.min(obs.iloc[:, i])
        maxobs = centerobs + slope * (maxsim - centersim)
        minobs = centerobs - slope * (centersim - minsim)

        msim = np.array([minsim, maxsim])
        mobs = np.array([minobs, maxobs])
        quant_sim = np.array([quant_1_sim, quant_3_sim])
        quant_obs = np.array([quant_1_obs, quant_3_obs])

        fig = plt.figure(figsize=fig_size, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111) 
        
        if not legend:
            plt.plot(sim_perc, obs_perc, linestyle[0], markersize=2)
            plt.plot(msim, mobs, linestyle[1], linewidth = linewidth[0])
            plt.plot(quant_sim, quant_obs, linestyle[2], marker='o', markerfacecolor='k', linewidth = linewidth[1])
        else:
            plt.plot(sim_perc, obs_perc, linestyle[0], markersize=2, label = q_labels[0])
            plt.plot(msim, mobs, linestyle[1], linewidth = linewidth[0], label = q_labels[1])
            plt.plot(quant_sim, quant_obs, linestyle[2], marker='o', markerfacecolor='w', linewidth = linewidth[1], label = q_labels[2])

        # Placing Labels, title and grid if requested
        plt.xticks(fontsize=15, rotation=45)
        plt.yticks(fontsize=15)

        if labels:
            # Plotting Labels
            plt.xlabel(labels[0], fontsize=18)
            plt.ylabel(labels[1]+" (m\u00B3/s)", fontsize=18)
        
        if title:
            title_dict = {'family': 'sans-serif',
                        'color': 'black',
                        'weight': 'normal',
                        'size': 20,
                        }
            ## Check that the title is a list of strings or a single string
            if isinstance(title, list):
                ax.set_title(title[i] if i < len(title) else f"QQ Plot {i + 1}", fontdict=title_dict, pad=25)
            elif isinstance(title, str):
                ax.set_title(label=title, fontdict=title_dict, pad=25)

        # Placing a grid if requested
        if grid:
            plt.grid(True)

        # Save or auto-save for large column counts
        auto_save = len(obs.columns) > 5
        _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "qqplot")  


    return
