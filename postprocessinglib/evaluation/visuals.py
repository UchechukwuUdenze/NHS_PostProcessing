"""
The visual module contains different plotting functions for time series visualization.
It allows users to plot hydrographs per station for each stations to allow us visualize
the time-series data. These graphs provide a simple and clear way to immeditely identify
patterns and discrepancies with model operation.
They are also made to very customizable with a lot of options to suit the need of many types
of users.   
Some of them also allow their metrics to be placed beside the plots as shown below:

.. image:: Figures/Visuals.png
  :alt: graphs showing graph types
.. image:: Figures/Visuals_m.png
  :alt: line plot showing metrics

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
from postprocessinglib.utilities import _helper_functions as hlp

def _save_or_display_plot(fig, save: bool, save_as: Union[str, List[str]], dir: str, i: int, type: str):
    """
    Save the plot to a file or display it based on user preferences.

    This helper function determines whether to save the plot to a specified directory or display 
    it on the screen. If saving, the plot is saved as a PNG file with a specified or default filename.
    If not saving, the plot is displayed interactively using Matplotlib's `plt.show()`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure instance to be saved or displayed.

    save : bool
        Whether to save the plot to a file. If `True`, the plot is saved. If `False`, the plot is displayed.

    save_as : str or list of str
        The name or list of names to save the plot as. If provided, the plot is saved with this name(s). 
        If `save_as` is a list, the plot is saved using the corresponding name for each figure. Default is None.

    dir : str
        The directory where the plot will be saved. If the directory does not exist, it will be created. 
        Default is the current working directory.

    i : int
        The index for generating unique filenames when saving multiple plots. Used when `save_as` is a list.

    type : str
        The type of the plot (e.g., 'scatter-plot'). This is used to generate a default filename if 
        `save_as` is not provided.

    Returns
    -------
    None
        This function does not return anything. It either saves or displays the plot.

    Example
    -------
    Save a plot with a custom filename:

    >>> fig = plt.figure()
    >>> # Plotting code...
    >>> _save_or_display_plot(fig, save=True, save_as="my_plot", dir="./plots", i=0, type="scatter")

    Display a plot:

    >>> _save_or_display_plot(fig, save=False, save_as="my_plot", dir="./plots", i=0, type="scatter")

    Notes
    -----
    - If `save_as` is a string, the plot will be saved with that name.
    - If `save_as` is a list, the plot will be saved with the corresponding name from the list, 
      using the index `i` to select the correct filename.
    - The `plt.tight_layout()` function is called to ensure the plot layout is adjusted before saving.

    """
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

def _finalize_plot(ax, grid, labels, title, name, i):
    """
    Finalizes the plot by setting labels, title, and grid options.
    """
    ax.legend(fontsize=15)
    
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=15)

    if labels:
        plt.xlabel(labels[0], fontsize=18)
        plt.ylabel(labels[1]+" (m\u00B3/s)", fontsize=18)

    if title:
        title_dict = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 20}
        if isinstance(title, list):
            ax.set_title(title[i] if i < len(title) else f"{name}_{i + 1}", fontdict=title_dict, pad=25)
        elif isinstance(title, str):
            ax.set_title(label=title, fontdict=title_dict, pad=25)

    if grid:
        plt.grid(True) 

def plot(
    merged_df: pd.DataFrame = None, 
    df: pd.DataFrame = None, 
    sim_df: pd.DataFrame = None,
    legend: tuple[str, str] = ('Data',), 
    metrices: list[str] = None,
    grid: bool = False, 
    title: str = None, 
    labels: list[str] = None, 
    padding: bool = False,
    linestyles: tuple[str, str] = ('r-',), 
    linewidth: tuple[float, float] = (1.5,),
    fig_size: tuple[float, float] = (10, 6), 
    metrics_adjust: tuple[float, float] = (1.05, 0.5),
    plot_adjust: float = 0.2, 
    save: bool = False, 
    save_as: str = None, 
    dir: str = os.getcwd()
    ) -> plt.figure:
    """ Create a comparison time series line plot of simulated and observed time series data

    This function generates line plots for any number of observed and simulated data
    
    The function can handle data provided in three formats:
    - A merged DataFrame containing both observed and simulated data.
    - A Single DataFrame of your choosing.
    - A DataFrame containing only simulated data.

    The plot allows customization of various visual elements like line style, colors, axis labels, and title. 
    The resulting figure can be displayed or saved to a specified directory and file name.

    Parameters
    ----------
    merged_df : pd.DataFrame, optional
        The dataframe containing the series of observed and simulated values. It must have a datetime index.
        To be use when the data contains both observed and simulated values.

    sim_df : pd.DataFrame, optional
        A DataFrame containing only the simulated data series

    df : pd.DataFrame, optional
        A Single DataFrame usually containing only one of either simulated or observed data... or any data.

    legend : tuple of str, optional
        A tuple containing the labels for the data being plotted

    metrices : list of str, optional
        A list of metrics to display on the plot, default is None.

    grid : bool, optional
        Whether to display a grid on the plot, default is False.

    title : str, optional
        The title of the plot.

    labels : list of strs, optional
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
                    num_sim = num_sim,
                    title='Simulated vs Observed',
                    labels=['Time', 'Value'], grid=True,
                    metrices = ['KGE','RMSE'])

    .. image:: ../Figures/plot1_example.png

    >>> # Example 2: Plotting only observed and simulated data with custom linestyles and saving the plot
    >>> obs_data = pd.DataFrame({...})  # Your observed data
    >>> sim_data = pd.DataFrame({...})  # Your simulated data
    >>> visuals.plot(obs_df = obs_data, sim_df = sim_data, linestyles=('g-', 'b-'), num_sim = num_sim,
                    save=True, save_as="plot2_example", dir="../Figures")

    .. image:: ../Figures/plot2_example.png

    >>> # Example 3: Plotting a single dataframe
    >>> single_data = pd.DataFrame({...})  # Your single dataframe (either simulated or observed)
    >>> visuals.plot(df=single_data, grid=True, title="Single Line Plot", labels=("Time", "Value"))

    .. image:: ../Figures/plot3_example.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_

    Notes
    -----
    - The function requires at least one valid data input (merged_df, sim_df, or df).
    - The time index of the input DataFrames must be a datetime index or convertible to datetime.
    - If the number of columns in the `obs_df` or `sim_df` exceeds five, the plot will be automatically saved.
    - Metrics will be displayed on the plot if specified in the `metrices` parameter.
         
    """
    if df is None:
        # Get the number of simulated data columns
        num_sim = sum(1 for col in  merged_df.columns if col[0] == merged_df.columns[0][0])-1 if merged_df is not None else sum(1 for col in  sim_df.columns if col[0] == sim_df.columns[0][0])
        print(f"Number of simulated data columns: {num_sim}")
        # Line width generation
        if len(linewidth) < num_sim + 1:
            print("Number of linewidths provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of linewidths provided is: ", str(len(linewidth)) +
                    ". Defaulting to 1.5")
            linewidth = linewidth + (1.5,) * (num_sim + 1 if merged_df is not None else num_sim)
        
        # Generate colors dynamically using Matplotlib colormap
        cmap = plt.cm.get_cmap("tab10", num_sim + 1)  # +1 for Observed
        colors = [cmap(i) for i in range(num_sim + 1)]

        # Available line styles
        # base_linestyles = ["-", "--", "-.", ":"]
        style = ('-',) * (num_sim + 1) # default to solid lines unless overwritten

        # Generate linestyles dynamically
        if len(linestyles) < num_sim + 1:
            print("Number of linestyles provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of linestyles provided is: ", str(len(linestyles)) +
                    ". Defaulting to solid lines (-)")
            linestyles = linestyles + tuple(f"{colors[i % len(colors)]}{style[i % len(style)]}" 
                            for i in range(num_sim + 1 if merged_df is not None else num_sim))
            
        # Generate Legends dynamically
        if len(legend) < num_sim + 1:
            print("Number of legends provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of legends provided is: ", str(len(legend)) +
                    ". Applying Default legend names")
            legend = (["Observed"] + [f"Simulated {i+1}" for i in range(num_sim)] if merged_df is not None else [f"Simulated {i+1}" for i in range(num_sim)])           
            

    # Assign the data based on inputs
    sims = {}
    obs = None
    if merged_df is not None:
        # If merged_df is provided, separate observed and simulated data
        obs = merged_df.iloc[:, ::num_sim+1]
        for i in range(1, num_sim+1):
            sims[f"sim_{i}"] = merged_df.iloc[:, i::num_sim+1]
        time = merged_df.index
    elif sim_df is not None:
        # If sim_df is provided, that means theres no observed.
        for i in range(0, num_sim):
            sims[f"sim_{i+1}"] = sim_df.iloc[:, i::num_sim]
        time = sim_df.index
    elif df is not None:
        # If only df is provided, it could be either obs, simulated or just random data.
        # obs = df # to keep the future for loop valid
        line_df = df
        time = df.index
    else:
        raise RuntimeError('Please provide valid data (merged_df, sim_df, or df)')

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
                # weekly
                # datetime ignores the week specifier unless theres a weekday attached,
                # so we extrct the week number and attach Monday - day 1
                time = [pd.to_datetime(f"{int(float(week))}-W{int((float(week) - int(float(week))) * 100):02d}-4", format="%Y-W%U-%w") for week in time]
            elif '-' in time[0]:
                # monthly
                time = [pd.to_datetime(f"{month}-15") for month in time]
            else: # yearly
                time = [pd.to_datetime(f"{year}-07-15") for year in time]
    
    if df is not None:
        for i in range (0, len(line_df.columns)):
            # Plotting the Data     
            fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')
            ax.plot(time, line_df.iloc[:, i], linestyles[0], label=legend[0], linewidth = linewidth[0])

            if padding:
                plt.xlim(time[0], time[-1])
            _finalize_plot(ax, grid, labels, title, "plot", i)
            auto_save = len(sims["sim_1"].columns) > 5
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "plot")
    else:
        # In either case of merged or sim_df, we will alwaays have simulated data, so we plot the obs first if we have it.
        for i in range (0, len(sims["sim_1"].columns)):
            # Plotting the Data     
            fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')
            if obs is not None:                
                ax.plot(time, obs.iloc[:, i], color = eval(linestyles[0][:-1]) if linestyles[0][:-1].startswith("(") else linestyles[0][:-1], 
                        linestyle = linestyles[0][-1],label=legend[0], linewidth = linewidth[0])
            for j in range(1, num_sim+1):
                ax.plot(time, sims[f"sim_{j}"].iloc[:, i], color = eval(linestyles[j][:-1]) if linestyles[j][:-1].startswith("(") else linestyles[j][:-1],
                        linestyle = linestyles[j][-1], label=legend[j], linewidth = linewidth[j])            
            if padding:
                plt.xlim(time[0], time[-1])
            _finalize_plot(ax, grid, labels, title, "plot", i)

            # Placing Metrics on the Plot if requested
            # if metrices:
            #     formatted_selected_metrics = 'Metrics:\n'
            #     if df is None:
            #         if metrices == 'all':
            #             for key, value in metrics.calculate_all_metrics(observed=obs, simulated=sim).items():
            #                 formatted_selected_metrics += key + ' : ' + str(value[i]) + '\n'
            #         else: 
            #             assert isinstance(metrices, list)
            #             for key, value in metrics.calculate_metrics(observed=obs, simulated=sim, metrices=metrices).items():
            #                 formatted_selected_metrics += key + ' : ' + str(value[i]) + '\n'


            #     font = {'family': 'sans-serif',
            #             'weight': 'normal',
            #             'size': 12}
            #     plt.text(metrics_adjust[0], metrics_adjust[1], formatted_selected_metrics, ha='left',
            #             va='center', transform=ax.transAxes, fontdict=font, #mouseover = True,
            #             bbox = dict(boxstyle = "round, pad = 0.5,rounding_size=0.3", facecolor = "0.8", edgecolor="k"))

            #     plt.subplots_adjust(right = 1-plot_adjust)

            # Save or auto-save for large column counts
            auto_save = len(sims["sim_1"].columns) > 5
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
    Generate a bounded time-series plot comparing observed and simulated streamflow with confidence intervals.

    A bounded plot is a time-series visualization that compares observed and simulated hydrological data while incorporating confidence bounds to represent uncertainty.
    This function plots the streamflow data against Julian days, providing insights into seasonal variations and model performance over time. 
    The confidence bounds, which can be defined using minimum-maximum ranges or percentiles (e.g., 5th-95th or 25th-75th percentiles), highlight the range of variability in the observed and simulated datasets. 
    The function allows for flexible customization of labels, legends, transparency, and line styles. 
    This visualization is particularly useful for evaluating hydrological models, identifying systematic biases, and assessing the reliability of simulated streamflow under different flow conditions. 

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

    >>> # Create an index for the data
    >>> time_index = pd.date_range(start='2025-01-01', periods=50, freq='D')
    >>> # Generate sample observed and simulated data
    >>> obs_data = pd.DataFrame({
    ...     "Station1_Observed": np.random.rand(50),
    ...     "Station2_Observed": np.random.rand(50)
    ... }, index=time_index)
    >>> sim_data = pd.DataFrame({
    ...     "Station1_Simulated": np.random.rand(50),
    ...     "Station2_Simulated": np.random.rand(50)
    ... }, index=time_index)

    >>> # Combine observed and simulated data
    >>> data = pd.concat([obs_data, sim_data], axis=1)
    >>> # Generate sample bounds
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

    >>> # Plot the data with bounds
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

    >>> # Adjust a few other metrics
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

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_
    
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

            # Prepare bounds for the current column
            upper_obs = _prepare_bounds(upper_bounds, i, observed=True)
            lower_obs = _prepare_bounds(lower_bounds, i, observed=True)
            upper_sim = _prepare_bounds(upper_bounds, i, observed=False)
            lower_sim = _prepare_bounds(lower_bounds, i, observed=False)

            # Plot bounds
            for j in range(len(upper_bounds)):
                ax.fill_between(time, lower_obs[j], upper_obs[j], alpha=transparency[1], color=linestyles[1][0])
                ax.fill_between(time, lower_sim[j], upper_sim[j], alpha=transparency[0], color=linestyles[0][0])
            
            if padding:
                plt.xlim(time[0], time[-1])
            _finalize_plot(ax, grid, labels, title, "bounded-plot", i)

            # Save or auto-save for large column counts
            auto_save = len(line_obs.columns) > 5
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "bounded-plot")
        
# def bounded_plot(
#     lines: Union[List[pd.DataFrame], pd.DataFrame],
#     upper_bounds: List[pd.DataFrame] = None,
#     lower_bounds: List[pd.DataFrame] = None,
#     legend: Tuple[str, str] = ('Simulated Data', 'Observed Data'),
#     grid: bool = False,
#     title: Union[str, List[str]] = None,
#     labels: Tuple[str, str] = None,
#     linestyles: Tuple[str, str] = ('r-', 'b-'),
#     padding: bool = False,
#     fig_size: Tuple[float, float] = (10, 6),
#     transparency: Tuple[float, float] = (0.4, 0.4),
#     save: bool = False,
#     save_as: Union[str, List[str]] = None,
#     dir: str = os.getcwd()
#     ) -> plt.figure:
#     """ 
#     Plots time-series data with optional confidence bounds.
#     Generate a bounded time-series plot comparing observed and simulated streamflow with confidence intervals.

#     A bounded plot is a time-series visualization that compares observed and simulated hydrological data while incorporating confidence bounds to represent uncertainty.
#     This function plots the streamflow data against Julian days, providing insights into seasonal variations and model performance over time. 
#     The confidence bounds, which can be defined using minimum-maximum ranges or percentiles (e.g., 5th-95th or 25th-75th percentiles), highlight the range of variability in the observed and simulated datasets. 
#     The function allows for flexible customization of labels, legends, transparency, and line styles. 
#     This visualization is particularly useful for evaluating hydrological models, identifying systematic biases, and assessing the reliability of simulated streamflow under different flow conditions. 

#     Parameters
#     ----------
#     lines : list of pd.DataFrame
#         A list of DataFrames containing the observed and simulated data series to be plotted. Each DataFrame must have a datetime index.

#     upper_bounds : list of pd.DataFrame, optional
#         A list of DataFrames containing the upper bounds for each series. If not provided, no upper bounds are plotted.

#     lower_bounds : list of pd.DataFrame, optional
#         A list of DataFrames containing the lower bounds for each series. If not provided, no lower bounds are plotted.

#     legend : tuple of str, optional
#         A tuple containing the labels for the simulated and observed data, default is ('Simulated Data', 'Observed Data').

#     grid : bool, optional
#         Whether to display a grid on the plot, default is False.

#     title : str, optional
#         The title of the plot.

#     labels : tuple of str, optional
#         A tuple containing the labels for the x and y axes.

#     linestyles : tuple of str, optional
#         A tuple specifying the line styles for the simulated and observed data.

#     padding : bool, optional
#         Whether to add padding to the x-axis limits for a tighter plot, default is False.

#     fig_size : tuple of float, optional
#         A tuple specifying the size of the figure.

#     transparency : list of float, optional
#         A list specifying the transparency levels for the upper and lower bounds, default is [0.4, 0.4].

#     save : bool, optional
#         Whether to save the plot to a file, default is False.

#     save_as : str or list of str, optional
#         The name or list of names to save the plot as. If a list is provided, each plot will be saved with the corresponding name.

#     dir : str, optional
#         The directory to save the plot to, default is the current working directory.

#     Returns
#     -------
#     fig : Matplotlib figure instance
    
#     Example
#     -------
#     Generate a bounded plot with simulated and observed data, along with upper and lower bounds.

#     >>> import pandas as pd
#     >>> import numpy as np
#     >>> from postprocessinglib.evaluation import visuals

#     >>> # Create an index for the data
#     >>> time_index = pd.date_range(start='2025-01-01', periods=50, freq='D')
#     >>> # Generate sample observed and simulated data
#     >>> obs_data = pd.DataFrame({
#     ...     "Station1_Observed": np.random.rand(50),
#     ...     "Station2_Observed": np.random.rand(50)
#     ... }, index=time_index)
#     >>> sim_data = pd.DataFrame({
#     ...     "Station1_Simulated": np.random.rand(50),
#     ...     "Station2_Simulated": np.random.rand(50)
#     ... }, index=time_index)

#     >>> # Combine observed and simulated data
#     >>> data = pd.concat([obs_data, sim_data], axis=1)
#     >>> # Generate sample bounds
#     >>> upper_bounds = [
#     ...     pd.DataFrame({
#     ...         "Station1_Upper": np.random.rand(50) + 0.5,
#     ...         "Station2_Upper": np.random.rand(50) + 0.5
#     ...     }, index=time_index)
#     ... ]
#     >>> lower_bounds = [
#     ...     pd.DataFrame({
#     ...         "Station1_Lower": np.random.rand(50) - 0.5,
#     ...         "Station2_Lower": np.random.rand(50) - 0.5
#     ...     }, index=time_index)
#     ... ]

#     >>> # Plot the data with bounds
#     >>> visuals.bounded_plot(
#     ...     lines=data,
#     ...     upper_bounds=upper_bounds,
#     ...     lower_bounds=lower_bounds,
#     ...     legend=('Simulated Data', 'Observed Data'),
#     ...     labels=('Datetime', 'Streamflow'),
#     ...     transparency = [0.4, 0.3],
#     ...     grid=True,
#     ...     save=True,
#     ...     save_as = 'bounded_plot_example',
#     ...     dir = '../Figures'
#     ... )

#     .. image:: ../Figures/bounded_plot_example_1.png

#     >>> # Adjust a few other metrics
#     >>> visuals.bounded_plot(
#     ...     lines = merged_df,
#     ...     upper_bounds = upper_bounds,
#     ...     lower_bounds = lower_bounds,
#     ...     title=['Long Term Aggregation by days of the Year'],
#     ...     legend = ['Predicted Streamflow','Recorded Streamflow'],
#     ...     linestyles=['k', 'r-'],
#     ...     labels=['Days of the year', 'Streamflow Values'],
#     ...     transparency = [0.4, 0.7],
#     ... )

#     .. image:: ../Figures/bounded_plot_example_2.png

#     `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_
    
#     """

#     ## Check that the inputs are DataFrames
#     if isinstance(lines, pd.DataFrame):
#         lines = [lines]
#     elif not isinstance(lines, list):
#         raise ValueError("Argument must be a dataframe or a list of dataframes.")
    
#     upper_bounds = upper_bounds or []
#     lower_bounds = lower_bounds or []

#     if not isinstance(upper_bounds, list) or not isinstance(lower_bounds, list):
#         raise ValueError("Bounds must be lists of DataFrames.")
#     if len(upper_bounds) != len(lower_bounds):
#         raise ValueError("Upper and lower bounds lists must have the same length.")

#     # Plotting
#     for line in lines:
#         if not isinstance(line, pd.DataFrame):
#             raise ValueError("All items in the 'lines' must be a DataFrame.")

#         # Get the number of simulated data columns
#         num_sim = sum(1 for col in  line.columns if col[0] == line.columns[0][0])-1 #if line is not None else sum(1 for col in  sim_df.columns if col[0] == sim_df.columns[0][0])
#         print(f"Number of simulated data columns: {num_sim}")
        
#         # Setting Variable for the simulated data, observed data, and time stamps
#         time = line.index
#         obs_cols = [col for col in line.columns if "QOMEAS" in col[1].upper()]
#         sim_cols = [col for col in line.columns if "QOSIM" in col[1].upper()]


#         for i in range (0, len(obs_cols) or 1):
#             fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')

#             if obs_cols:
#                 # Plot observed data
#                 line_obs = line[obs_cols]
#                 ax.plot(time, line_obs.iloc[:, i], 'k-', label=legend[1], linewidth = 1.5) # if theres observed, itll be a default black line

#             # Plot all simulations
#             for j, sim_col in enumerate(sim_cols):
#                 label = f"Sim {j + 1}"
#                 style = linestyles[j % len(linestyles)]
#                 ax.plot(time, line[sim_col], style, label=label, linewidth=1.5)

#             # Prepare bounds for the current column
#             upper_obs = _prepare_bounds(upper_bounds, i, observed=True)
#             lower_obs = _prepare_bounds(lower_bounds, i, observed=True)
#             upper_sim = _prepare_bounds(upper_bounds, i, observed=False)
#             lower_sim = _prepare_bounds(lower_bounds, i, observed=False)

#             # Plot bounds
#             for j in range(len(upper_bounds)):
#                 ax.fill_between(time, lower_obs[j], upper_obs[j], alpha=transparency[1], color=linestyles[1][0])
#                 ax.fill_between(time, lower_sim[j], upper_sim[j], alpha=transparency[0], color=linestyles[0][0])
            
#             if padding:
#                 plt.xlim(time[0], time[-1])
#             _finalize_plot(ax, grid, labels, title, "bounded-plot", i)

#             # Save or auto-save for large column counts
#             auto_save = len(line_obs.columns) > 5
#             _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "bounded-plot")        

def histogram(
    merged_df: pd.DataFrame = None, 
    df: pd.DataFrame = None, 
    obs_df: pd.DataFrame = None, 
    sim_df: pd.DataFrame = None,
    bins: int = 100,
    legend: Tuple[str, str] = ('Simulated Data', 'Observed Data'),
    colors: Tuple[str, str] = ('r', 'b'),
    transparency: float = 0.6,
    z_norm=False,
    prob_dens=False,
    fig_size: Tuple[float, float] = (12, 6),
    title: str = None,
    labels: Tuple[str, str] = ('Value', 'Frequency'),
    grid: bool = False,
    save: bool = False,
    save_as: str = None,
    dir: str = os.getcwd()
    ) -> plt.figure:
    """
    Plots Histogram for Observed and Simulated Data with Optional Normalization

    This function generates a histogram comparing the distribution of observed and simulated data, providing insights into their statistical characteristics and variability.
    The histogram allows users to analyze the frequency distribution of hydrological data, assess model performance, and identify biases in the simulated dataset.
    The function supports Z-score normalization, which transforms the data into standard deviations from the mean, enabling comparison of datasets with different scales. 
    It also includes an option to plot the histogram as a probability density function (PDF), ensuring that the area under the histogram sums to one, making it easier to compare distributions.
    Users can customize the number of bins, colors, legend labels, and transparency levels to enhance visualization clarity. The function also allows for gridlines, axis labeling,
    and automatic or manual saving of plots.
    This visualization is particularly useful for hydrological modeling, statistical analysis, and understanding deviations between observed and simulated streamflow distributions under various conditions.

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

    bins: int
        Specifies the number of bins in the histogram.

    z_norm: bool
        If True, the data will be Z-score normalized.
    
    prob_dens: bool
        If True, normalizes both histograms to form a probability density, i.e., the area
        (or integral) under each histogram will sum to 1.

    legend: tuple of str
        Tuple of length two with str inputs. Adds a Legend in the 'best' location determined by
        matplotlib. The entries in the tuple label the simulated and observed data
        (e.g. ['Simulated Data', 'Predicted Data']).

    grid: bool
        If True, adds a grid to the plot.

    title: str
        If given, sets the title of the plot.

    labels: tuple of str
        Tuple of two string type objects to set the x-axis labels and y-axis labels, respectively.

    figsize: tuple of float
        Tuple of length two that specifies the horizontal and vertical lengths of the plot in
        inches, respectively.

    colors : tuple of str, optional
        Colors for the simulated and observed histograms.

    transparency : float, optional
        Transparency level for the histograms, default is 0.6.

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
                    bins = 100,
                    labels=['Frequency', 'Value'], grid=True)

    .. image:: ../Figures/hist1_Example.png

    >>> # Example 2: Plotting observed and simulated data with custom linestyles and saving the plot
    >>> obs_data = pd.DataFrame({...})  # Your observed data
    >>> sim_data = pd.DataFrame({...})  # Your simulated data
    >>> visuals.plot(obs_df = obs_data, sim_df = sim_data, colors=('g', 'c'), bins = 100, z_norm = True, prob_dens = True,
                    save=True, save_as="hist2_example", dir="../Figures")

    .. image:: ../Figures/hist2_Example.png

    >>> # Example 3: Plotting a single dataframe
    >>> single_data = pd.DataFrame({...})  # Your single dataframe (either simulated or observed)
    >>> visuals.plot(df=single_data, grid=True, title="Single Histogram Plot", labels=("Time", "Frequency"))

    .. image:: ../Figures/hist3_Example.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_

    """
    # Assign the data based on inputs
    if merged_df is not None:
        # If merged_df is provided, separate observed and simulated data
        obs = merged_df.iloc[:, ::2]
        sim = merged_df.iloc[:, 1::2]
    elif sim_df is not None and obs_df is not None:
        # If both sim_df and obs_df are provided
        obs = obs_df
        sim = sim_df
    elif df is not None:
        # If only df is provided, treat it as both observed and simulated data
        obs = df # to keep the future for loop valid
        sim = None
        line_df = df
    else:
        raise RuntimeError('Please provide valid data (merged_df, obs_df, sim_df, or df)')
    
    for i in range (0, len(obs.columns)):
        # Manipulating and generating the Data
        if z_norm:
            # calculating the z-score for the observed data
            obs.iloc[:, i] = (obs.iloc[:, i] - obs.iloc[:, i].mean()) / obs.iloc[:, i].std()

            if sim is not None:
                # calculating the z-score for the simulated data 
                sim.iloc[:, i] = (sim.iloc[:, i] - sim.iloc[:, i].mean()) / sim.iloc[:, i].std()

        # finding the mimimum and maximum z-scores
        total_max = max(obs.iloc[:, i].max(), sim.iloc[:, i].max()) if sim is not None else obs.iloc[:, i].max()
        total_min = min(obs.iloc[:, i].min(), sim.iloc[:, i].min()) if sim is not None else obs.iloc[:, i].min()
        num_bins = np.linspace(total_min - 0.01, total_max + 0.01, bins)

        # creating the bins based on the max and min
        num_bins = np.linspace(total_min - 0.01, total_max + 0.01, bins)    

        # Getting the fig and axis handles
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111) 

        # Plotting the Data
        ax.hist(obs.iloc[:, i],
                bins=num_bins,
                alpha=transparency,
                label=legend[1],
                color=colors[1],
                edgecolor='black',
                linewidth=0.5,
                density=prob_dens)
        if sim is not None:
            ax.hist(sim.iloc[:, i],
                bins=num_bins,
                alpha=transparency,
                label=legend[0],
                color=colors[0],
                edgecolor='black',
                linewidth=0.5,
                density=prob_dens)
            plt.legend(labels=[legend[1],legend[0]], loc='best')

        _finalize_plot(ax, grid, labels, title, "histogram", i)

        # Save or auto-save for large column counts
        auto_save = len(obs.columns) > 5
        _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "histogram")

def scatter(
  grid: bool = False, 
  title: str = None, 
  legend: tuple[str, str] = None,
  labels: tuple[str, str] = ('Simulated Data', 'Observed Data'),
  fig_size: tuple[float, float] = (10, 6), 
  best_fit: bool = False, 
  line45: bool = False,

  merged_df: pd.DataFrame = None, 
  obs_df: pd.DataFrame = None, 
  sim_df: pd.DataFrame = None,
  metrices: list[str] = None, 
  markerstyle: list[str] = ['bo'], 
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
  simulated: pd.DataFrame = None,
  cmap: str='jet',
  vmin: float=None,
  vmax:float=None
  ) -> plt.figure:
    """
    Creates a scatter plot comparing observed and simulated data, with optional features like 
    best fit lines, 45-degree reference lines, and metric annotations.

    This function can handle both merged data (observed and simulated in a single DataFrame) and 
    separate observed and simulated data DataFrames. Additionally, it can plot scatter plots over 
    shapefiles for geographic data visualization.

    The plot can be customized with various visual features, such as the color map, gridlines, 
    markers, and axis labels. The function also allows adding a linear regression best-fit line, 
    a 45-degree line, and annotations for metrics. The plot can be saved to a file if desired.

    Parameters
    ----------
    grid : bool, optional
        Whether to display a grid on the plot, default is False.

    title : str, optional
        The title of the plot.

    labels: tuple of str, optional
        A tuple containing the labels for the simulated and observed data, default is ('Simulated Data', 'Observed Data').

    legend : tuple of str, optional
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
    
    cmap: string, optional
        Used to determine the color scheme of the color map for the shapefile plot 

    vmin: float, optional
        Minimum colormap value
    
    vmax: float, optional
        Maximum colormap value
    
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
    >>>     grid=True,
    >>>     metrices = ['KGE','RMSE'],
    >>>     line45=True,
    >>>     markerstyle = 'b.',
    >>>     metrices = ['KGE','RMSE']
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
                        cmap = 'jet'
                    )

    .. image:: ../Figures/SRB_subDrainage_showing_KGE.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_

    """
    # Plotting the Data
    if not shapefile_path:
        # Get the number of simulated data columns
        num_sim = sum(1 for col in  merged_df.columns if col[0] == merged_df.columns[0][0])-1 if merged_df is not None else sum(1 for col in  sim_df.columns if col[0] == sim_df.columns[0][0])
        print(f"Number of simulated data columns: {num_sim}")

        # Generate colors dynamically using Matplotlib colormap
        color_map = plt.cm.get_cmap("tab10", num_sim)  # +1 for Observed
        colors = [color_map(i) for i in range(num_sim)]

        # Available marker styles
        style = [".", "1", "v", "x", "*", "+", "X", "3", "^", "s", "D"] # default unless overwritten

        # Generate linestyles dynamically
        if len(markerstyle) < num_sim:
            print("Number of markerstyles provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim) + ". Number of markerstyles provided is: ", str(len(markerstyle)) +
                    ". Using Default Markerstyles.")
            markerstyle = markerstyle + [f"{colors[i % len(colors)]}{style[i % len(style)]}" 
                            for i in range(num_sim)]    

        # Generate Legends dynamically
        if legend is None:
            legend = [f"Simulated {i}" for i in range(1, num_sim+1)]
        elif len(legend) < num_sim:
            print("Number of legends provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim) + ". Number of legends provided is: ", str(len(legend)) +
                    ". Applying Default labels")
            legend = legend + [f"Simulated {len(legend)+i}" for i in range(1, num_sim+1)]

        sims = {}
        obs = None
        if merged_df is not None:
            # If merged_df is provided, separate observed and simulated data
            obs = merged_df.iloc[:, ::num_sim+1]
            for i in range(1, num_sim+1):
                sims[f"sim_{i}"] = merged_df.iloc[:, i::num_sim+1]
        elif sim_df is not None and obs_df is not None:
            # If both sim_df and obs_df are provided
            obs = obs_df
            for i in range(0, num_sim):
                sims[f"sim_{i+1}"] = sim_df.iloc[:, i::num_sim]
        else:
            raise RuntimeError('Please provide valid data (merged_df, obs_df or sim_df)')

        for i in range (0, len(obs.columns)):
            max_obs = obs.iloc[:, i].max()
            min_obs = obs.iloc[:, i].min()
            max_sim, min_sim  = 0, 0

            # Plotting the Data
            fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')
            for j in range(1, num_sim+1):
                ax.plot(sims[f"sim_{j}"].iloc[:, i], obs.iloc[:, i],
                        color = eval(markerstyle[j-1][:-1]) if markerstyle[j-1][:-1].startswith("(") else markerstyle[j-1][:-1],
                        marker = markerstyle[j-1][-1], 
                        label=legend[j-1] if labels else f"Sim {j}", 
                        linestyle='None')
                max_sim = np.max([max_sim, sims[f"sim_{j}"].iloc[:, i].max()])
                min_sim = np.min([min_sim, sims[f"sim_{j}"].iloc[:, i].min()]) 

                if best_fit:
                    # Getting a polynomial fit and defining a function with it
                    p = np.polyfit(sims[f"sim_{j}"].iloc[:, i], obs.iloc[:, i], 1)
                    f = np.poly1d(p)

                    # Calculating new x's and y's
                    x_new = np.linspace(sims[f"sim_{j}"].iloc[:, i].min(), sims[f"sim_{j}"].iloc[:, i].max(), sims[f"sim_{j}"].iloc[:, i].size)
                    y_new = f(x_new)

                    # Formatting the best fit equation to be able to display in latex
                    equation = "{} x + {}".format(np.round(p[0], 4), np.round(p[1], 4))

                    # Plotting the best fit line with the equation as a legend in latex
                    ax.plot(x_new, y_new,
                            color = eval(markerstyle[j-1][:-1]) if markerstyle[j-1][:-1].startswith("(") else markerstyle[j-1][:-1], 
                            label="${}$".format(equation))

            
            if line45:
                max = np.nanmax([max_sim, max_obs])
                min = np.nanmin([min_sim, min_obs])
                # Plotting the 45 degree line
                ax.plot(np.arange(int(min), int(max) + 1), np.arange(int(min), int(max) + 1), 'r--', label='45$^\u00b0$ Line')

            
            if best_fit or line45:
                ax.legend(fontsize=12)
            
            _finalize_plot(ax, grid, labels, title, "scatter-plot", i)               

            # Placing Metrics on the Plot if requested
            # if metrices:
            #     formatted_selected_metrics = 'Metrics: \n'
            #     if metrices == 'all':
            #         for key, value in metrics.calculate_all_metrics(observed=obs, simulated=sim).items():
            #             formatted_selected_metrics += key + ' : ' + str(value[i]) + '\n'
            #     else: 
            #         assert isinstance(metrices, list)
            #         for key, value in metrics.calculate_metrics(observed=obs, simulated=sim, metrices=metrices).items():
            #             formatted_selected_metrics += key + ' : ' + str(value[i]) + '\n'

            #     font = {'family': 'sans-serif',
            #             'weight': 'normal',
            #             'size': 12}
            #     plt.text(metrics_adjust[0], metrics_adjust[1], formatted_selected_metrics, ha='left',
            #         va='center', transform=ax.transAxes, fontdict=font, #mouseover = True,
            #         bbox = dict(boxstyle = "round4, pad = 0.6,rounding_size=0.3", facecolor = "0.8", edgecolor="k"))

            #     plt.subplots_adjust(right = 1-plot_adjust)

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
        sc = gdf_points.plot(ax=ax, column=list(metr)[0], cmap=cmap, vmin = vmin, vmax=vmax,legend=True, markersize=40, legend_kwds={'label': list(metr)[0]+" Value", "orientation": "vertical"})

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
        
        # Save or auto-save for large column counts
        _save_or_display_plot(fig, save, save_as, dir, i=0, type="shapefile-plot")


def qqplot(
    grid: bool = False, 
    title: str = None, 
    labels: tuple[str, str] = None, 
    fig_size: tuple[float, float] = (10, 6),
    method: str = "linear", 
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
    """
    Generate a Quantile-Quantile (QQ) plot to compare the statistical distribution of simulated and observed data.

    A Quantile-Quantile (QQ) plot is a graphical technique for assessing whether two datasets come from the same distribution by plotting their quantiles against each other. 
    If the datasets have identical distributions, the points should fall along the 1:1 line. This function calculates and visualizes the quantiles of observed and simulated streamflow data, interpolating if necessary, and marks key statistical features such as the interquartile range. 
    By comparing the empirical quantiles of simulated and observed data, the QQ plot helps evaluate the performance of hydrological models in reproducing streamflow distributions, highlighting potential biases and differences in variability.
    The function also allows for flexible customization of labels, legends, transparency, and line styles.
    It is an essential tool in hydrology and environmental sciences for assessing the agreement between measured and modeled hydrological variables.

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

    method: str
        Determines whether the quantiles should be interpolated when the data length differs.
        If True, the quantiles are interpolated to align the data lengths between the observed
        and simulated data, ensuring accurate comparison.
        Default is Linear.

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
        Labels for the x-axis (simulated quantiles) and y-axis (observed quantiles).
        Default is ['Quantiles', 'Range of Quantiles', 'Inter Quartile Range'].

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
    >>> index = pd.date_range(start="2022-01-01", periods=50, freq="D")
    >>> obs_df = pd.DataFrame({
    >>>     "Station1": np.random.rand(50),
    >>>     "Station2": np.random.rand(50)
    >>> }, index=index)
    >>> #
    >>> sim_df = pd.DataFrame({
    >>>     "Station1": np.random.rand(50),
    >>>     "Station2": np.random.rand(50)
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

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_

    """

    # Get the number of simulated data columns
    num_sim = sum(1 for col in  merged_df.columns if col[0] == merged_df.columns[0][0])-1 if merged_df is not None else sum(1 for col in  sim_df.columns if col[0] == sim_df.columns[0][0])
    print(f"Number of simulated data columns: {num_sim}")

    # Generate colors dynamically using Matplotlib colormap
    color_map = plt.cm.get_cmap("tab10", num_sim)  # +1 for Observed
    colors = [color_map(i) for i in range(num_sim)]

    # Available marker styles
    style = [".", "1", "v", "x", "*", "+", "X", "3", "^", "s", "D"] # default unless overwritten

    # Generate linestyles dynamically
    if len(markerstyle) < num_sim:
        print("Number of markerstyles provided is less than the number of columns. "
                "Number of columns : " + str(num_sim) + ". Number of markerstyles provided is: ", str(len(markerstyle)) +
                ". Using Default Markerstyles.")
        markerstyle = markerstyle + [f"{colors[i % len(colors)]}{style[i % len(style)]}" 
                        for i in range(num_sim)]    

    # Generate Legends dynamically
    if legend is None:
        legend = [f"Simulated {i}" for i in range(1, num_sim+1)]
    elif len(legend) < num_sim:
        print("Number of legends provided is less than the number of columns. "
                "Number of columns : " + str(num_sim) + ". Number of legends provided is: ", str(len(legend)) +
                ". Applying Default labels")
        legend = legend + [f"Simulated {len(legend)+i}" for i in range(1, num_sim+1)]

    sims = {}
    obs = None
    if merged_df is not None:
        # If merged_df is provided, separate observed and simulated data
        obs = merged_df.iloc[:, ::num_sim+1]
        for i in range(1, num_sim+1):
            sims[f"sim_{i}"] = merged_df.iloc[:, i::num_sim+1]
    elif sim_df is not None and obs_df is not None:
        # If both sim_df and obs_df are provided
        obs = obs_df
        for i in range(0, num_sim):
            sims[f"sim_{i+1}"] = sim_df.iloc[:, i::num_sim]
    else:
        raise RuntimeError('Please provide valid data (merged_df, obs_df or sim_df)')

    for i in range (0, len(obs.columns)):
        fig = plt.figure(figsize=fig_size, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)

        n = obs.iloc[:, i].size
        pvec = 100 * ((np.arange(1, n + 1) - 0.5) / n)
        obs_perc = np.percentile(obs.iloc[:, i], pvec, method=method)
        # Finding the interquartile range to plot the best fit line
        quant_1_obs = np.percentile(obs.iloc[:, i], quantile[0], interpolation=method)
        quant_3_obs = np.percentile(obs.iloc[:, i], quantile[1], interpolation=method)
        dobs = quant_3_obs - quant_1_obs

        for j in range(1, num_sim+1):
            sim_perc = np.percentile(sims[f"sim_{j}"].iloc[:, i], pvec, method=method)
            # Finding the interquartile range to plot the best fit line
            quant_1_sim = np.percentile(sims[f"sim_{j}"].iloc[:, i], quantile[0], interpolation=method)
            quant_3_sim = np.percentile(sims[f"sim_{j}"].iloc[:, i], quantile[1], interpolation=method)
            dsim = quant_3_sim - quant_1_sim

            slope = dobs / dsim

            centersim = (quant_1_sim + quant_3_sim) / 2
            centerobs = (quant_1_obs + quant_3_obs) / 2
            maxsim = np.max(sims[f"sim_{j}"].iloc[:, i])
            minsim = np.min(sims[f"sim_{j}"].iloc[:, i])
            maxobs = centerobs + slope * (maxsim - centersim)
            minobs = centerobs - slope * (centersim - minsim)

            msim = np.array([minsim, maxsim])
            mobs = np.array([minobs, maxobs])
            quant_sim = np.array([quant_1_sim, quant_3_sim])
            quant_obs = np.array([quant_1_obs, quant_3_obs]) 
        

            plt.plot(sim_perc, obs_perc, linestyle[0],  label = q_labels[0], markersize=2)
            plt.plot(msim, mobs, linestyle[1], label = q_labels[1], linewidth = linewidth[0])
            plt.plot(quant_sim, quant_obs, linestyle[2], label = q_labels[2], marker='o', markerfacecolor='w', linewidth = linewidth[1])
            plt.legend(fontsize=12)

            _finalize_plot(ax, grid, labels, title, "qqplot", i)

            # Save or auto-save for large column counts
            auto_save = len(obs.columns) > 5
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "qqplot")  


def flow_duration_curve(
    merged_df: pd.DataFrame = None, 
    sim_df: pd.DataFrame = None,
    df: pd.DataFrame = None, 
    legend: tuple[str, str] = ('Data',), 
    grid: bool = False, 
    title: str = None, 
    labels: tuple[str, str] = ('Exceedance Probability (%)', 'Flow'),
    linestyles: tuple[str, str] = ('r-',), 
    linewidth: tuple[float, float] = (1.5,),
    fig_size: tuple[float, float] = (10, 6), 
    save: bool = False, 
    save_as: str = None, 
    dir: str = os.getcwd()
) -> plt.figure:
    """
    Generate a Flow Duration Curve (FDC) comparing observed and simulated streamflow.
    
    A Flow Duration Curve (FDC) is a graphical representation of the percentage of time that streamflow is equal to or exceeds a particular value over a given period. 
    It provides insights into the variability and availability of water in a river system, capturing both high and low flow conditions. 
    This function calculates the exceedance probability of observed and simulated streamflow, ranks the values from highest to lowest, and plots them on a probability scale.
    The function allows for flexible customization of labels, legends, transparency, and line styles. 
    The FDC is a crucial tool in hydrology for assessing water availability, evaluating hydrological model performance, and understanding flow regime characteristics.

    
    Parameters
    ----------
    merged_df : pd.DataFrame, optional
        A DataFrame containing both observed and simulated streamflow data. The observed data should be 
        in the even-numbered columns, and the simulated data in the odd-numbered columns.
        
    sim_df : pd.DataFrame, optional
        A DataFrame containing simulated streamflow data. This is used if `merged_df` is not provided.
        
    legend : tuple of str, optional
        A tuple with two string labels for the legend: the first for the simulated data and the second 
        for the observed data. Defaults to ('Simulated Data', 'Observed Data').

    grid : bool, optional
        Whether to display a grid on the plot. Defaults to False.

    title : str, optional
        Title of the plot. If not provided, no title will be displayed.

    labels : tuple of str, optional
        A tuple with two string labels for the x and y axes. Defaults to 
        ('Exceedance Probability (%)', 'Flow (m³/s)').

    linestyles : tuple of str, optional
        A tuple with two strings specifying the line styles for the simulated and observed data, respectively. 
        Defaults to ('r-', 'b-').

    linewidth : tuple of float, optional
        A tuple with two floats specifying the line widths for the simulated and observed data, respectively. 
        Defaults to (1.5, 1.25).

    fig_size : tuple of float, optional
        A tuple with two floats specifying the width and height of the figure in inches. Defaults to (10, 6).

    save : bool, optional
        Whether to save the plot as a file. Defaults to False.

    save_as : str, optional
        The file name (with extension) to save the plot as. Only used if `save=True`.

    dir : str, optional
        The directory to save the plot file. Only used if `save=True`. Defaults to the current working directory.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure instance containing the FDC plot.

    Raises
    ------
    RuntimeError
        If neither `merged_df` nor both `obs_df` and `sim_df` are provided.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import postprocessinglib.evaluation.visuals as visuals
    >>> # Example observed and simulated data
    >>> observed = pd.DataFrame(np.random.randn(100, 1), columns=["Flow"])
    >>> simulated = pd.DataFrame(np.random.randn(100, 1), columns=["Flow"])
    >>> visuals.flow_duration_curve(observed=observed, simulated=simulated, title="FDC Example", grid =True)

    .. image:: ../Figures/FDC_Example.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_
    """

    if df is None:
        # Get the number of simulated data columns
        num_sim = sum(1 for col in  merged_df.columns if col[0] == merged_df.columns[0][0])-1 if merged_df is not None else sum(1 for col in  sim_df.columns if col[0] == sim_df.columns[0][0])
        print(f"Number of simulated data columns: {num_sim}")
        # Line width generation
        if len(linewidth) < num_sim + 1:
            print("Number of linewidths provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of linewidths provided is: ", str(len(linewidth)) +
                    ". Defaulting to 1.5")
            linewidth = linewidth + (1.5,) * (num_sim + 1 if merged_df is not None else num_sim)
        
        # Generate colors dynamically using Matplotlib colormap
        cmap = plt.cm.get_cmap("tab10", num_sim + 1)  # +1 for Observed
        colors = [cmap(i) for i in range(num_sim + 1)]

        # Available line styles
        # base_linestyles = ["-", "--", "-.", ":"]
        style = ('-',) * (num_sim + 1) # default to solid lines unless overwritten

        # Generate linestyles dynamically
        if len(linestyles) < num_sim + 1:
            print("Number of linestyles provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of linestyles provided is: ", str(len(linestyles)) +
                    ". Defaulting to solid lines (-)")
            linestyles = linestyles + tuple(f"{colors[i % len(colors)]}{style[i % len(style)]}" 
                            for i in range(num_sim + 1 if merged_df is not None else num_sim))
            
        # Generate Legends dynamically
        if len(legend) < num_sim + 1:
            print("Number of legends provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of legends provided is: ", str(len(legend)) +
                    ". Applying Default legend names")
            legend = (["Observed"] + [f"Simulated {i+1}" for i in range(num_sim)] if merged_df is not None else [f"Simulated {i+1}" for i in range(num_sim)]) 
    
    
    # Assign the data based on inputs
    sims = {}
    obs = None
    if merged_df is not None:
        # If merged_df is provided, separate observed and simulated data
        obs = merged_df.iloc[:, ::num_sim+1]
        for i in range(1, num_sim+1):
            sims[f"sim_{i}"] = merged_df.iloc[:, i::num_sim+1]
        time = merged_df.index
    elif sim_df is not None:
        # If sim_df is provided, that means theres no observed.
        for i in range(0, num_sim):
            sims[f"sim_{i+1}"] = sim_df.iloc[:, i::num_sim]
        time = sim_df.index
    elif df is not None:
        # If only df is provided, it could be either obs, simulated or just random data.
        # obs = df # to keep the future for loop valid
        line_df = df
        time = df.index
    else:
        raise RuntimeError('Please provide valid data (merged_df, sim_df, or df)')

    if df is not None:
        for i in range (0, len(line_df.columns)):
            # Plotting the Data     
            fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')
            line_df_sorted = np.sort(line_df.iloc[:, i])[::-1]
            exceedance_prob = np.linspace(0, 100, len(line_df_sorted))
            ax.plot(exceedance_prob, line_df_sorted, linestyles[0], label=legend[0], linewidth=linewidth[0])

            _finalize_plot(ax, grid, labels, title, "fdc-plot", i)
            
            # Save or auto-save for large column counts
            auto_save = len(obs.columns) > 5
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "fdc-plot") 
    else:
        # In either case of merged or sim_df, we will alwaays have simulated data, so we plot the obs first if we have it.
        for i in range (0, len(sims["sim_1"].columns)):
            # Plotting the Data     
            fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')
            if obs is not None:
                obs_sorted = np.sort(obs.iloc[:, i])[::-1]
                exceedance_prob = np.linspace(0, 100, len(obs_sorted))                
                ax.plot(exceedance_prob, obs_sorted, color = eval(linestyles[0][:-1]) if linestyles[0][:-1].startswith("(") else linestyles[0][:-1], 
                        linestyle = linestyles[0][-1],label=legend[0], linewidth = linewidth[0])
            for j in range(1, num_sim+1):
                sim_sorted = np.sort(sims[f"sim_{j}"].iloc[:, i])[::-1]  # Sorting the simulated data
                exceedance_prob = np.linspace(0, 100, len(obs_sorted))
                ax.plot(exceedance_prob, sim_sorted, color = eval(linestyles[j][:-1]) if linestyles[j][:-1].startswith("(") else linestyles[j][:-1],
                        linestyle = linestyles[j][-1], label=legend[j], linewidth = linewidth[j])            

            _finalize_plot(ax, grid, labels, title, "fdc-plot", i)
    
            # Save or auto-save for large column counts
            auto_save = len(obs.columns) > 5
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "fdc-plot") 


