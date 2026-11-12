"""Plotting helpers for Problems 5-7 (NumPy vs. PyTorch loss curves, learning curves per regulariser,
validation-loss comparison)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .mlp_torch import AGREEMENT_TOLERANCE

if TYPE_CHECKING:
    from .mlp_numpy import TrainingLog
    from .regularisation import RunResult

# Same tokens as list03: categorical slots in fixed order, ink and grid greys.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
TRAIN_COLOR, VAL_COLOR = SERIES_COLORS[0], SERIES_COLORS[1]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def _style_axes(ax: Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelcolor=INK_SECONDARY)


def plot_numpy_vs_torch(log_np: TrainingLog, log_torch: TrainingLog, steps_per_epoch: int) -> Figure:
    """Loss curves of the two implementations on top of each other, and their difference (Problem 5).

    Parameters
    ----------
    log_np, log_torch : TrainingLog
        Histories of the NumPy and the PyTorch run (same weights, batches and learning rate).
    steps_per_epoch : int
        Mini-batches per epoch, to put the per-step losses on the epoch axis.

    Returns
    -------
    Figure
        Left: mini-batch and full training-set loss of both runs (log scale). Right: absolute
        difference per step; exact zeros cannot be drawn on a log axis and are counted in the title.
    """
    fig, (ax_loss, ax_diff) = plt.subplots(1, 2, figsize=(13, 4.3), constrained_layout=True)
    steps = np.arange(1, log_np.step_loss.size + 1) / steps_per_epoch
    epochs = np.arange(log_np.epoch_loss.size)

    ax_loss.plot(steps, log_np.step_loss, color=TRAIN_COLOR, linewidth=0.8, alpha=0.35)
    ax_loss.plot(steps, log_torch.step_loss, color=VAL_COLOR, linewidth=0.8, alpha=0.35, linestyle=(0, (3, 3)))
    ax_loss.plot(epochs, log_np.epoch_loss, color=TRAIN_COLOR, linewidth=2, label="NumPy")
    ax_loss.plot(epochs, log_torch.epoch_loss, color=VAL_COLOR, linewidth=2, linestyle="--", label="PyTorch")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy (log scale)")
    ax_loss.set_title("Training loss: thin = mini-batch, thick = full training set", color=INK)
    ax_loss.legend(frameon=False)
    _style_axes(ax_loss)

    diff = np.abs(log_np.step_loss - log_torch.step_loss)
    nonzero = diff > 0
    ax_diff.scatter(steps[nonzero], diff[nonzero], s=6, color=TRAIN_COLOR, alpha=0.6, linewidths=0)
    ax_diff.axhline(AGREEMENT_TOLERANCE, color=INK_MUTED, linestyle="--", linewidth=0.9)
    ax_diff.text(steps[-1], AGREEMENT_TOLERANCE * 1.6, f"tolerance {AGREEMENT_TOLERANCE:g}",
                 ha="right", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    ax_diff.axhline(np.finfo(np.float64).eps, color=BASELINE, linestyle=":", linewidth=0.9)
    ax_diff.text(steps[-1], np.finfo(np.float64).eps * 1.6, "float64 machine epsilon",
                 ha="right", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    ax_diff.set_yscale("log")
    ax_diff.set_ylim(1e-18, 1e-8)
    ax_diff.set_xlabel("epoch")
    ax_diff.set_ylabel("|loss NumPy - loss PyTorch|")
    ax_diff.set_title(f"Per-step difference ({(~nonzero).sum()} of {diff.size} steps exactly equal)", color=INK)
    _style_axes(ax_diff)
    return fig


def plot_learning_curves(study: dict[str, list[RunResult]], seed_index: int = 0) -> Figure:
    """Train and validation loss and accuracy per configuration, one column each (Problem 6).

    Parameters
    ----------
    study : dict of str to list of RunResult
        Output of :func:`src.regularisation.run_study`.
    seed_index : int, default 0
        Which seed's run to draw (curves of single runs keep the real early-stopping point).

    Returns
    -------
    Figure
        Top row: cross-entropy on a log scale. Bottom row: accuracy. The dotted line marks
        the epoch of minimum validation loss; for early stopping it is the kept epoch.
    """
    names = list(study)
    fig, axes = plt.subplots(2, len(names), figsize=(3.6 * len(names), 6.4), sharex=True,
                             sharey="row", constrained_layout=True)
    for col, name in enumerate(names):
        run = study[name][seed_index]
        h = run.history
        best = int(h.loc[h["val_loss"].idxmin(), "epoch"])
        ax_loss, ax_acc = axes[0, col], axes[1, col]
        ax_loss.plot(h["epoch"], h["train_loss"], color=TRAIN_COLOR, linewidth=1.6)
        ax_loss.plot(h["epoch"], h["val_loss"], color=VAL_COLOR, linewidth=1.6)
        ax_acc.plot(h["epoch"], h["train_acc"], color=TRAIN_COLOR, linewidth=1.6)
        ax_acc.plot(h["epoch"], h["val_acc"], color=VAL_COLOR, linewidth=1.6)
        for ax in (ax_loss, ax_acc):
            ax.axvline(best, color=INK_MUTED, linestyle=":", linewidth=1)
            _style_axes(ax)
        right = best > 0.6 * h["epoch"].max()
        ax_loss.text(best, 0.98, f"min val loss \nepoch {best} " if right else f" min val loss\n epoch {best}",
                     transform=ax_loss.get_xaxis_transform(), va="top", ha="right" if right else "left",
                     fontsize=8, color=INK_SECONDARY)
        ax_loss.set_title(name, color=INK, fontsize=10.5)
        ax_acc.set_xlabel("epoch")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("cross-entropy (log scale)")
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylim(0.9, 1.005)
    handles = [Line2D([], [], color=TRAIN_COLOR, linewidth=2, label="train"),
               Line2D([], [], color=VAL_COLOR, linewidth=2, label="validation")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    return fig


def plot_val_loss_comparison(
    study: dict[str, list[RunResult]],
    start_epoch: int = 10,
    ylim: tuple[float, float] = (0.08, 0.22),
    ax: Axes | None = None,
) -> Figure:
    """Mean validation loss over seeds for every configuration on one axis (Problems 6-7).

    Each curve is averaged over the seeds and drawn up to the shortest run, so the
    early-stopping curves end where their first seed stopped. Early-stopping runs share
    the seeds of the runs without it, so they are dashed to stay visible on top of them.
    The first ``start_epoch`` epochs are cut and the y-axis zoomed, because the
    differences between configurations are small next to the initial drop.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    else:
        fig = ax.figure
    for i, (name, runs) in enumerate(study.items()):
        length = min(len(r.history) for r in runs)
        mean = np.mean([r.history["val_loss"].to_numpy()[:length] for r in runs], axis=0)
        epochs = np.arange(1, length + 1)
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        style = "--" if runs[0].config.early_stopping else "-"
        ax.plot(epochs, mean, color=color, linewidth=2, linestyle=style, label=name)
        j = int(np.argmin(mean))
        ax.plot(epochs[j], mean[j], "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=2)
    ax.set_xlim(start_epoch, None)
    ax.set_ylim(*ylim)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation cross-entropy")
    ax.set_title("Validation loss, mean over seeds (dot = minimum)", color=INK)
    ax.legend(frameon=False)
    _style_axes(ax)
    return fig
