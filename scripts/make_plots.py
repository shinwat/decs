import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import stats

def get_ci(data, confidence=0.95):

    medians = []
    lowers = []
    uppers = []
    for i in range(len(data[:,0])): # iterate through episodes
        policydata = data[i,:]
        median = np.median(policydata)
        res = stats.bootstrap((policydata,), np.median, confidence_level=confidence, method='basic')
        lower, upper = res.confidence_interval
        medians.append(median)
        lowers.append(lower)
        uppers.append(upper)
    return np.array((medians,lowers,uppers))

def plot_multiple_with_ci(
    data_list,
    *,
    x_values=None,
    line_colors=None,
    alpha=0.2,
    labels=None,
    line_kw=None,
    ci_kw=None,
    ylabel="Success Rate",
    legend='best',
):
    """
    Plot multiple lines with confidence intervals as shaded regions.

    Parameters
    ----------
    data_list : sequence of np.ndarray
        Each element is a 2D array of shape (3, n_steps), where:
            data[0, :] = values
            data[1, :] = lower CI (absolute values)
            data[2, :] = upper CI (absolute values)

    x_values : array-like, optional
        Values for the x-axis (must have length n_steps).
        If None, uses np.arange(n_steps).

    line_colors : sequence of color specs, optional
        Colors for each line. If None, uses matplotlib default cycle.

    alpha : float, optional
        Transparency for all CI shaded regions. Default is 0.2.

    labels : sequence of str, optional
        Labels for each line (for legend). If None, no labels are set.

    line_kw : dict, optional
        Extra keyword arguments passed to `ax.plot` for all lines.
        (You can override per-line by explicitly calling `ax` afterwards if needed.)

    ci_kw : dict, optional
        Extra keyword arguments passed to `ax.fill_between` for all CIs.

    title : str, optional
        Title of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if line_kw is None:
        line_kw = {}
    if ci_kw is None:
        ci_kw = {}

    # Ensure all arrays are consistent and get n_steps from the first one
    data_list = [np.asarray(d) for d in data_list]
    if not data_list:
        raise ValueError("`data_list` must contain at least one array.")

    n_steps = data_list[0].shape[1]
    for i, d in enumerate(data_list):
        if d.shape[0] != 3:
            raise ValueError(
                f"All arrays must have shape (3, n_steps). "
                f"Array {i} has shape {d.shape}."
            )
        if d.shape[1] != n_steps:
            raise ValueError(
                "All arrays must have the same number of steps. "
                f"Array 0 has {n_steps}, array {i} has {d.shape[1]}."
            )

    # X-axis
    if x_values is None:
        x = np.arange(n_steps)
    else:
        x = np.asarray(x_values)
        if x.shape[0] != n_steps:
            raise ValueError(
                f"`x_values` must have length {n_steps}. Got {x.shape[0]}."
            )

    # Colors
    if line_colors is None:
        # Let matplotlib cycle automatically by passing no color for some lines
        line_colors = [None] * len(data_list)
    elif len(line_colors) != len(data_list):
        raise ValueError(
            "`line_colors` must have the same length as `data_list`."
        )

    # Labels
    if labels is not None and len(labels) != len(data_list):
        raise ValueError("`labels` must have the same length as `data_list`.")

    fig, ax = plt.subplots()

    for i, (data, color) in enumerate(zip(data_list, line_colors)):
        values = data[0, :]
        ci_low = data[1, :]
        ci_high = data[2, :]

        # Choose label if provided
        label = labels[i] if labels is not None else None

        # Plot main line
        line_kwargs = dict(line_kw)  # shallow copy
        if color is not None:
            line_kwargs.setdefault("color", color)
        if label is not None:
            line_kwargs.setdefault("label", label)

        ax.plot(x, values, **line_kwargs)

        # Plot CI (use same color if given, otherwise matplotlib chooses)
        ci_kwargs = dict(ci_kw)
        if color is not None:
            ci_kwargs.setdefault("color", color)

        ax.fill_between(
            x,
            ci_low,
            ci_high,
            alpha=alpha,
            **ci_kwargs,
        )

    ax.set_xlabel("Episodes")
    ax.set_ylabel(ylabel)
    ax.set_title("Push Task Evaluation")

    if labels is not None:
        ax.legend(loc=legend)

    return fig, ax

def make_plots(directory, confidence, ylabel=None, legend=None):
    results = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            data = np.genfromtxt(os.path.join(root,file), delimiter=',', skip_header=0) # load csv
            results.append(get_ci(data, confidence))
            labels.append(file[:file.index("_")])
    episodes = np.linspace(0,1000,results[0][0].size)
    labels = None if len(labels) < 2 else labels
    # sort the label alphabetically and keep the original indexes
    labels_with_index = sorted((name, index) for index, name in enumerate(labels))
    # unzip
    labels, sortedIndex = zip(*labels_with_index)
    # sort the results according to the indexes
    results = [results[i] for i in sortedIndex]
    # print(results)
    fig, ax = plot_multiple_with_ci(
        results,
        x_values=episodes,
        labels=labels,
        ylabel=ylabel,
        legend=legend,
    )
    plt.show()

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help='The directory containing the csv files.'
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help='The confidence level for the boostrap method.'
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default=None,
        help='The Y label for the plot.'
    )
    parser.add_argument(
        "--legend",
        type=str,
        default=None,
        help='The location of the legend.'
    )
    args = parser.parse_args()
    
    make_plots(args.directory, args.confidence, args.ylabel, args.legend)