import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def plot_ccdf(df, var, group_col, top_labels=None, save_path=None, loglog=True):
    """
    Plot CCDF of a variable grouped by a community or topic.

    Parameters:
        df (pd.DataFrame): input data
        var (str): column to plot
        group_col (str): grouping column (e.g., 'S_modularity', 'label')
        top_labels (list): optional subset of groups to plot
        save_path (str): optional path to save plot
        loglog (bool): log-log axis scaling
    """
    labels = sorted(df[group_col].dropna().unique())
    if top_labels:
        labels = [label for label in labels if label in top_labels]

    stats = (df
             .melt(id_vars=var, value_vars=[group_col])
             .groupby([var, 'value'])
             .size()
             .unstack(fill_value=0))

    # Normalize, compute CDF and CCDF
    for col in stats.columns:
        stats[col] = stats[col] / stats[col].sum()
        stats[col] = 1 - stats[col].cumsum()

    stats = stats.reset_index()

    # Trim based on signal threshold
    min_vals = stats[labels].min(axis=1)
    threshold_index = min_vals[min_vals > 0.01].index
    if len(threshold_index) > 0:
        stats = stats.loc[:threshold_index[-1]]

    plt.figure(figsize=(8, 6))
    for label in labels:
        plt.semilogy(stats[var], stats[label], label=label)

    plt.xlabel(f"{var} (number of exposures)")
    plt.ylabel("CCDF (% Retweets)")
    plt.legend(title=group_col, loc="upper right")

    if loglog:
        plt.xscale('log')
        plt.yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sliding_window(df, time_col, label_col, window_days=7, step_days=1, save_path=None):
    """
    Plot timeline of retweet counts per label in sliding windows.

    Parameters:
        df (DataFrame): must contain datetime column and label column
        time_col (str): name of time column
        label_col (str): label (e.g., topic) column
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col)

    start = df[time_col].min()
    stop = df[time_col].max()

    times, labels = [], []
    i = 0
    while start + pd.Timedelta(days=window_days) <= stop:
        end = start + pd.Timedelta(days=window_days)
        temp = df[(df[time_col] >= start) & (df[time_col] < end)]
        times.extend([i] * len(temp))
        labels.extend(temp[label_col])
        start += pd.Timedelta(days=step_days)
        i += 1

    df_slide = pd.DataFrame({label_col: labels, 'window': times})
    dft = pd.crosstab(df_slide['window'], df_slide[label_col])

    ax = dft.plot(figsize=(12, 6))
    ax.set_xlabel("Sliding time window")
    ax.set_ylabel("Number of retweets")
    ax.legend(title=label_col, loc='upper left')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_reinforcement(df, group_col, var='sawl', time_col='time', save_path=None):
    """
    Plot reinforcement: min-mean-max exposure by group over time.

    Parameters:
        df (DataFrame): input data
        group_col (str): grouping column (e.g., S_modularity)
        var (str): exposure variable (e.g., sawl)
        time_col (str): datetime column
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    grouped = df.groupby([group_col, time_col])[var].agg(['min', 'mean', 'max']).unstack(group_col)

    fig, axes = grouped['mean'].plot(subplots=True, figsize=(12, 10), legend=False)
    palette = sns.color_palette()

    for idx, ax in enumerate(axes):
        group = grouped['mean'].columns[idx]
        ax.fill_between(grouped.index,
                        grouped['min'][group],
                        grouped['mean'][group],
                        color=palette[idx], alpha=0.2)
        ax.fill_between(grouped.index,
                        grouped['mean'][group],
                        grouped['max'][group],
                        color=palette[idx], alpha=0.2)
        ax.set_ylabel(group)
        ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_zoom_inset(x, y, zoom_xlim, zoom_ylim, zoom_factor=5, save_path=None):
    """
    Create zoomed-in inset plot.

    Parameters:
        x (array): x-axis data
        y (array): y-axis data
        zoom_xlim (tuple): x limits of zoomed region
        zoom_ylim (tuple): y limits of zoomed region
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    axins = zoomed_inset_axes(ax, zoom_factor, loc='upper right')
    axins.plot(x, y)
    axins.set_xlim(*zoom_xlim)
    axins.set_ylim(*zoom_ylim)

    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
