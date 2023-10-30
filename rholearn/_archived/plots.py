"""
Functions to buidl analysis plots using matplotlib
"""
from itertools import product
from typing import List, Dict, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import mpltex

import metatensor
from metatensor import TensorMap

from rholearn import error


def loss_vs_epoch(
    data: Union[List[Dict[int, np.ndarray]], List[np.ndarray]],
    mutliple_traces: bool = True,
    sharey: bool = False,
):
    """
    Takes arrays for loss values and returns a loglog plot of epoch vs loss.
    Assumes that the arrays passed for the losses correspond to a loss value for
    each epoch.

    If ``multiple_traces=True``, assumes the ``data`` dict passed will be of the
    form:

    {
        trace_0 <int>: losses_0 <np.ndarray>,
        trace_1 <int>: losses_1 <np.ndarray>,
        ...
    }

    and multiple traces will be plotted on the same axis. Otherwise if
    ``multiple_traces=False``, assumes the ``data`` will just be a numpy ndarray
    and a single trace will be plotted on the axis.

    If ``data`` is passed as a list of the above data formats, n horizontal
    subplots for each entry in the list will be created.
    """
    if not isinstance(data, list):
        data = [data]
    fig, ax = plt.subplots(1, len(data), sharey=sharey)
    for col, d in enumerate(data):
        linestyles = mpltex.linestyle_generator(markers=[])
        if mutliple_traces:
            subsets = np.sort(list(d.keys()))
            x = np.arange(len(d[0]))
            Y = [d[key] for key in subsets]
        else:
            x = np.arange(len(d))
            Y = [d]

        for i, y in enumerate(Y):
            ax[col].loglog(x, y, label=str(i), **next(linestyles))

        ax[col].set_xlabel("epoch")

    return fig, ax


def learning_curve(data, subset_sizes: np.array, point: str = "best"):
    """
    Plots the learning curves as a function of training subset size. From the
    array of loss values for each epoch either selects the "best" (i.e. lowest)
    or the "final" (i.e. the last epoch) loss value for each subset.

    Accepts as input a dict of the form:

    {
        trace_0 <int>: losses_0 <np.ndarray>, trace_1 <int>: losses_1
        <np.ndarray>, ...
    }

    or a list of these. If passing a list of these, a trace for each element
    will be plotted on the same plot.
    """
    if not isinstance(data, list):
        data = [data]
    fig, ax = plt.subplots()
    linestyles = mpltex.linestyle_generator(lines=["-"], hollow_styles=[])
    for col, d in enumerate(data):
        subsets = np.sort(list(d.keys()))
        x = subset_sizes
        if point == "final":
            y = [d[subset][-1] for subset in subsets]
        elif point == "best":
            y = [np.min(d[subset]) for subset in subsets]
        else:
            raise ValueError("``point`` must be 'final' or 'best'")
        if len(x) != len(y):
            raise ValueError(
                f"number of subset_sizes (x={x}) passed not equal to number of"
                + f" loss values gathered (y={y})."
            )
        ax.loglog(x, y, **next(linestyles))
    ax.set_xlabel(r"number of training structures")

    return fig, ax


def parity_plot(
    target: TensorMap,
    predicted: TensorMap,
    color_by: str = "spherical_harmonics_l",
):
    """
    Returns a parity plot of the target (x axis) and predicted (y axis)
    values. Plots also a grey dashed y=x line. The keys of the input TensorMap
    ``color_by`` decides what to colour the data by.
    """
    # Check that the metadata is equivalent between the 2 TensorMaps
    metatensor.equal_metadata(target, predicted)
    # Build the parity plot
    fig, ax = plt.subplots()
    linestyles = mpltex.linestyle_generator(
        lines=[], markers=["o"], hollow_styles=[False]
    )

    key_ranges = {key: np.unique(target.keys[key]) for key in target.keys.names}
    color_by_range = key_ranges.pop(color_by)
    key_names = list(key_ranges.keys())

    for color_by_idx in color_by_range:
        x, y = np.array([]), np.array([])

        for combo in list(product(*[key_ranges[key] for key in key_names])):
            # other_keys = {key_names[i]: combo[i] for i in range(len(key_names))}
            other_keys = {name_i: combo_i for name_i, combo_i in zip(key_names, combo)}
            try:
                target_block = target.block(**{color_by: color_by_idx}, **other_keys)
                pred_block = predicted.block(**{color_by: color_by_idx}, **other_keys)
            except ValueError as e:
                assert str(e).startswith(
                    "Couldn't find any block matching the selection"
                )
                print(f"key not found for {color_by} = {color_by_idx} and {other_keys}")
            try:
                x = np.append(x, target_block.values.detach().numpy().flatten())
                y = np.append(y, pred_block.values.detach().numpy().flatten())
            except AttributeError:
                x = np.append(x, target_block.values.flatten())
                y = np.append(y, pred_block.values.flatten())
        ax.plot(x, y, label=f"{color_by} = {color_by_idx}", **next(linestyles))
    # Plot a y=x grey dashed line
    ax.axline((-1e-5, -1e-5), (1e-5, 1e-5), color="grey", linestyle="dashed")
    fig.tight_layout()

    return fig, ax


def plot_coeff_hist(
    tensors: Union[TensorMap, List[TensorMap]],
    nbins: Optional[float] = 100,
    epsi: Optional[float] = None,
):
    """
    Plots the histogram of the absolute value of the coefficients of
    ``tensors``. The rows of the subplot correspond to the different L values,
    while the columns correspond to the different tensors passed.

    Histograms are plotted on a log-log scale, such that only coefficients that
    are greater than zero are plotted.

    Returns the matplotlib figure and axes, as well as a dict of the flattened
    coefficients for each l value.
    """
    # Check the input tensors
    if isinstance(tensors, TensorMap):
        tensors = [tensors]
    if len(tensors) > 1:
        for tensor in tensors[1:]:
            if not metatensor.equal_metadata(tensors[0], tensor):
                raise ValueError(
                    "metadata of all tensors must be equal to plot them together"
                )

    # Get the list of L values
    L = np.sort(np.unique(tensors[0].keys["spherical_harmonics_l"]))

    # Compile flattened vectors of absolute nonzero coefficients for each l
    # value
    results = [{l: np.array([]) for l in L} for tensor in tensors]
    for tensor_i, tensor in enumerate(tensors):
        for key, block in tensor.items():
            l = key["spherical_harmonics_l"]
            vals = block.values.flatten()
            results[tensor_i][l] = np.concatenate([results[tensor_i][l], vals])

    # Build the subplots
    fig, axes = plt.subplots(
        len(L),
        len(tensors),
        figsize=(4 * len(tensors), 3 * len(L)),
        sharex=True,
        sharey=True,
    )
    # Plot histograms with each l on a different row, and each tensor in a different column
    for tensor_i in range(len(tensors)):
        for l in L:
            # Take the absolute and nonzero values
            vals = np.abs(results[tensor_i][l])
            vals = vals[vals > 0]
            # Compute the bins sizes in log space
            logbins = np.logspace(np.log10(np.min(vals)), np.log10(np.max(vals)), nbins)
            # Plot and set axes to log scale
            if len(tensors) == 1:
                axes[l].hist(vals, bins=logbins, ec="k")
                axes[l].set_xscale("log")
                axes[l].set_yscale("log")
            else:
                axes[l, tensor_i].hist(vals, bins=logbins, ec="k")
                axes[l, tensor_i].set_xscale("log")
                axes[l, tensor_i].set_yscale("log")

    # Add row titles for the L values
    pad = 5
    row_titles = [f"L = {l}" for l in L]
    tmp_axes = axes if len(tensors) == 1 else axes[:, 0]
    for ax, row_title in zip(tmp_axes, row_titles):
        ax.annotate(
            row_title,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )
    # Add column titles
    if len(tensors) > 1:
        col_titles = [f"tensor {tensor_i}" for tensor_i in range(len(tensors))]
        for ax, col_title in zip(axes[0], col_titles):
            ax.annotate(
                col_title,
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                size="large",
                ha="center",
                va="baseline",
            )
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)

    return fig, ax, results


def plot_comparison_hist(
    a: TensorMap,
    b: TensorMap,
    nbins: Optional[float] = 100,
    epsi: Optional[float] = None,
):
    """
    Calcualtes the relative error between two TensorMaps ``a`` and ``b``, only
    for elements in both ``a`` and ``b`` that are greater than ``epsi``. Plots
    the values of ``a``, ``b``, and the relative error in a 3 column plot. Each
    row corresponds to a different l value.

    Returns the matplotlib figure and axes objects, and the results dict.
    """

    # First, calculate the relative errors
    results = error.relative_errors_a_b(a, b, epsi)

    # Get the list of L values
    L = np.sort(list(results.keys()))

    # Build the subplots
    fig, axes = plt.subplots(
        len(L), 3, figsize=(12, 3 * len(L)), sharex="col", sharey=True
    )
    for l in L:
        # First column - a values
        a_vals = results[l]["a_vals"]
        a_vals = a_vals[a_vals > 0]
        logbins_a = np.logspace(
            np.log10(np.min(a_vals)), np.log10(np.max(a_vals)), nbins
        )
        axes[l, 0].hist(a_vals, bins=logbins_a, ec="k")
        # Second column - b values
        b_vals = results[l]["b_vals"]
        b_vals = b_vals[b_vals > 0]
        logbins_b = np.logspace(
            np.log10(np.min(b_vals)), np.log10(np.max(b_vals)), nbins
        )
        axes[l, 1].hist(b_vals, bins=logbins_b, ec="k")
        # Third column - relative error
        err_vals = results[l]["rel_error"]
        err_vals = err_vals[err_vals > 0]
        if len(err_vals) > 0:
            logbins_err = np.logspace(
                np.log10(np.min(err_vals)), np.log10(np.max(err_vals)), nbins
            )
            axes[l, 2].hist(err_vals, bins=logbins_err, ec="k")
        # Plot epsi if passed
        if epsi is not None:
            axes[l, 0].axline((epsi, 0), (epsi, 1), color="grey", linestyle="dashed")
            axes[l, 1].axline((epsi, 0), (epsi, 1), color="grey", linestyle="dashed")

    # Set log scales
    for row in axes:
        for ax in row:
            ax.set_xscale("log")
            ax.set_yscale("log")

    # Add row titles for the L values
    pad = 5
    row_titles = [f"L = {l}" for l in L]
    for ax, row_title in zip(axes[:, 0], row_titles):
        ax.annotate(
            row_title,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )

    # Add column titles
    col_titles = ["|a|", "|b|", "relative error: | (a - b) / (a + b) |"]
    for ax, col_title in zip(axes[0], col_titles):
        ax.annotate(
            col_title,
            xy=(0.5, 1),
            xytext=(0, pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )

    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)

    return fig, ax, results


@mpltex.presentation_decorator
def save_fig_mpltex(fig, path_to_filename: str):
    """
    Uses mpltex (https://github.com/liuyxpp/mpltex) to save the matplotlib
    ``fig`` to file at ``path_to_filename`` (with no extension) with their
    decorator that produces a publication-quality figure as a pdf.
    """
    fig.savefig(path_to_filename)
