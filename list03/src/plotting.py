"""Plotting helpers for Problems 4-6 (confusion matrices, SGNS vs. PMI, analogy accuracy,
gender projections)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure

# Categorical slots in fixed order (slot 1 = first model, slot 2 = second model).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
# One-hue sequential ramp (blue, light -> dark) for magnitudes.
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def _style_axes(ax: Axes, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelcolor=INK_SECONDARY)


def plot_confusion_matrices(cms: dict[str, np.ndarray], class_names: list[str]) -> Figure:
    """Confusion matrices of several classifiers side by side (Problem 4).

    The colour is the row-normalised share (recall per true class), on one scale shared
    by all panels; each cell is annotated with its raw count.

    Parameters
    ----------
    cms : dict of str to np.ndarray
        Confusion matrices (rows = true class, columns = predicted), keyed by model.
    class_names : list of str
        Class names in label order.

    Returns
    -------
    Figure
        One panel per model with a shared colour bar.
    """
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    k = len(class_names)
    for ax, (name, cm) in zip(axes, cms.items()):
        share = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(share, cmap=BLUE_RAMP, vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                color = "#ffffff" if share[i, j] > 0.55 else INK
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9, color=color)
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{name} (accuracy {acc:.1%})", color=INK)
        ax.set_xticks(range(k), class_names, rotation=40, ha="right")
        ax.set_yticks(range(k), class_names if ax is axes[0] else [""] * k)
        ax.set_xlabel("predicted class", color=INK_SECONDARY)
        if ax is axes[0]:
            ax.set_ylabel("true class", color=INK_SECONDARY)
        ax.tick_params(length=0, labelcolor=INK_SECONDARY)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="share of the true class")
    return fig


def plot_pmi_vs_dot(pairs: pd.DataFrame, targets: tuple[str, ...]) -> Figure:
    """Density of trained SGNS dot products against shifted-PMI targets (Problems 3 and 5).

    Parameters
    ----------
    pairs : pd.DataFrame
        One row per word pair with a ``dot`` column and one column per target
        (``shifted_pmi_check``).
    targets : tuple of str
        Target columns, one panel each.

    Returns
    -------
    Figure
        Hexbin panels with the identity line ``dot = target`` and a shared colour bar.
    """
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.9), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    dot = pairs["dot"].to_numpy()
    lo = np.floor(min(dot.min(), *(pairs[t].min() for t in targets)))
    hi = np.ceil(max(dot.max(), *(pairs[t].max() for t in targets)))
    for ax, target in zip(axes, targets):
        x = pairs[target].to_numpy()
        hb = ax.hexbin(x, dot, gridsize=60, cmap=BLUE_RAMP, norm=LogNorm(), mincnt=1, extent=(lo, hi, lo, hi), linewidths=0)
        ax.plot([lo, hi], [lo, hi], color=INK_SECONDARY, linewidth=1)
        ax.text(hi - 0.2, hi - 0.6, "dot = target", ha="right", va="top", fontsize=9, color=INK_SECONDARY)
        r = np.corrcoef(x, dot)[0, 1]
        ax.set_title(f"{target}\nPearson r = {r:.2f}, mean(dot - target) = {np.mean(dot - x):+.2f}", fontsize=10, color=INK)
        ax.set_xlabel("target from co-occurrence counts", color=INK_SECONDARY)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        _style_axes(ax, grid_axis="both")
    axes[0].set_ylabel(r"trained dot product $u \cdot v$ (wv $\cdot$ syn1neg)", color=INK_SECONDARY)
    fig.colorbar(hb, ax=axes, shrink=0.85, label="word pairs per hexagon")
    return fig


def plot_analogy_accuracy(accuracy: pd.DataFrame, models: tuple[str, ...]) -> Figure:
    """Accuracy per analogy section on the questions every model covers (Problem 5).

    Parameters
    ----------
    accuracy : pd.DataFrame
        Output of ``analogy_accuracy`` (sections and totals as rows).
    models : tuple of str
        Model columns to draw, in palette order.

    Returns
    -------
    Figure
        Horizontal grouped bars; each label carries the number of common questions.
    """
    table = accuracy[accuracy["n_common"] > 0]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(table) + 1.4))
    y = np.arange(len(table))[::-1].astype(float)
    # A gap separates the 14 sections from the three totals.
    y[table["kind"].to_numpy() != "total"] += 1.0
    height = 0.8 / len(models)
    for m, name in enumerate(models):
        offset = (m - (len(models) - 1) / 2) * height
        ax.barh(y - offset, table[name], height=height * 0.9, color=SERIES_COLORS[m], label=name)
    totals = table["kind"].to_numpy() == "total"
    for m, name in enumerate(models):
        offset = (m - (len(models) - 1) / 2) * height
        for yy, value in zip(y[totals] - offset, table.loc[totals, name]):
            ax.text(value + 0.01, yy, f"{value:.1%}", va="center", fontsize=8, color=INK_SECONDARY)
    labels = [f"{s} (n = {n})" for s, n in zip(table.index, table["n_common"])]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("accuracy (top-1, 3CosAdd)", color=INK_SECONDARY)
    ax.set_title("Google analogy test set: questions covered by both models", color=INK)
    _style_axes(ax, grid_axis="x")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def plot_gender_projection(projection: pd.DataFrame) -> Figure:
    """Occupation words projected on each model's gender direction (Problem 6).

    One panel per model, aligned on the same occupations and the same horizontal scale.
    Positive values lean towards the male words of the definitional pairs (he, man, ...),
    negative values towards the female words.

    Parameters
    ----------
    projection : pd.DataFrame
        Output of ``occupation_projection`` (occupations x models); NaN = not in vocabulary.

    Returns
    -------
    Figure
        Lollipop panels sorted by the last model's projection, most female-associated at
        the bottom.
    """
    models = list(projection.columns)
    table = projection.loc[projection[models[-1]].sort_values().index]
    y = np.arange(len(table))
    limit = 1.15 * np.nanmax(np.abs(table.to_numpy()))
    fig, axes = plt.subplots(
        1, len(models), figsize=(4.4 * len(models) + 1.2, 0.3 * len(table) + 1.9), sharex=True, sharey=True
    )
    for m, (ax, name) in enumerate(zip(np.atleast_1d(axes), models)):
        values = table[name].to_numpy()
        known = ~np.isnan(values)
        ax.axvline(0, color=INK_SECONDARY, linewidth=1)
        ax.hlines(y[known], 0, values[known], color=SERIES_COLORS[m], linewidth=1.5)
        ax.scatter(values[known], y[known], s=42, color=SERIES_COLORS[m], zorder=3, edgecolors="#ffffff", linewidths=1)
        for yy in y[~known]:
            ax.text(0.03 * limit, yy, "not in vocabulary", va="center", fontsize=8, color=INK_MUTED)
        ax.set_xlim(-limit, limit)
        ax.set_title(name, color=INK, pad=20)
        ax.text(0.0, 1.01, "← female-associated", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=INK_SECONDARY)
        ax.text(1.0, 1.01, "male-associated →", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK_SECONDARY)
        ax.set_xlabel("cosine with the gender direction", color=INK_SECONDARY)
        _style_axes(ax, grid_axis="x")
    axes[0].set_yticks(y, table.index)
    fig.tight_layout()
    return fig
