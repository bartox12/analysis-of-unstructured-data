"""Plotting helpers for Problems 5-8 (modality overview, explained variance, 2-D embeddings by class,
structure-preservation diagnostics)."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .pca import n_components_for_variance

if TYPE_CHECKING:
    from .embeddings import EmbeddingResult


def plot_modalities(
    text: str,
    image: np.ndarray,
    audio: np.ndarray,
    sr: int,
    n_text_chars: int = 300,
) -> Figure:
    """Show a text preview, an image and an audio waveform side by side (Problem 5).

    Parameters
    ----------
    text : str
        Decoded text document; only the first ``n_text_chars`` characters are shown.
    image : np.ndarray
        Image array of shape ``(H, W)`` or ``(H, W, C)``.
    audio : np.ndarray
        Waveform of shape ``(n_samples,)`` or ``(n_channels, n_samples)``.
    sr : int
        Sampling rate of ``audio`` in Hz.
    n_text_chars : int, default 300
        Number of leading characters of ``text`` to render.

    Returns
    -------
    Figure
        Figure with three panels: text, image and waveform.
    """
    fig, (ax_text, ax_img, ax_audio) = plt.subplots(
        1, 3, figsize=(16, 4.5), gridspec_kw={"width_ratios": [1.2, 1, 1.4]}
    )

    preview = textwrap.fill(" ".join(text[:n_text_chars].split()), width=45)
    ax_text.text(
        0.0, 1.0, preview + " ...",
        va="top", ha="left", family="monospace", fontsize=8,
        transform=ax_text.transAxes,
    )
    ax_text.set_title(f"Text: first {n_text_chars} of {len(text)} characters")
    ax_text.axis("off")

    ax_img.imshow(image)
    ax_img.set_title(f"Image: shape {image.shape}, {image.dtype}")
    ax_img.axis("off")

    channels = np.atleast_2d(audio)
    n_samples = channels.shape[-1]
    t = np.arange(n_samples) / sr
    for ch in channels:
        ax_audio.plot(t, ch, linewidth=0.4)
    ax_audio.set_xlim(0, n_samples / sr)
    ax_audio.set_xlabel("time [s]")
    ax_audio.set_ylabel("amplitude")
    ax_audio.set_title(f"Audio: sr = {sr} Hz, duration = {n_samples / sr:.2f} s")

    fig.tight_layout()
    return fig


def plot_cumulative_explained_variance(
    evr: np.ndarray,
    thresholds: tuple[float, ...] = (0.8, 0.9, 0.95),
    ax: Axes | None = None,
) -> Figure:
    """Plot the cumulative explained-variance ratio against the number of components.

    Per-component ratios are drawn as light bars on the same (0-1) axis. Each
    threshold gets a dashed horizontal line and a vertical marker at the smallest
    ``k`` that reaches it, annotated with ``k``.

    Parameters
    ----------
    evr : np.ndarray
        Explained-variance ratios in decreasing order, shape ``(k,)``.
    thresholds : tuple of float, default (0.8, 0.9, 0.95)
        Variance fractions to mark.
    ax : Axes or None, default None
        Axes to draw on; a new figure is created if ``None``.

    Returns
    -------
    Figure
        The figure containing ``ax``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    evr = np.asarray(evr)
    k = np.arange(1, len(evr) + 1)
    cumulative = np.cumsum(evr)

    ax.bar(k, evr, width=0.8, color="tab:blue", alpha=0.3, label="per component")
    ax.step(k, cumulative, where="mid", color="tab:blue", linewidth=2, label="cumulative")

    for t in thresholds:
        k_t = n_components_for_variance(evr, t)
        ax.axhline(t, color="0.45", linestyle="--", linewidth=0.8)
        ax.axvline(k_t, ymax=t / 1.01, color="0.45", linestyle=":", linewidth=0.8)
        ax.plot(k_t, cumulative[k_t - 1], "o", color="tab:blue", markersize=5)
        ax.annotate(
            f"{t:.0%}: k = {k_t}", (k_t, t), xytext=(6, -14), textcoords="offset points",
            fontsize=9, color="0.2",
        )

    ax.set_xlim(0, len(evr) + 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("number of components k")
    ax.set_ylabel("explained variance ratio")
    ax.set_title("Cumulative explained variance")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="center right", frameon=False)
    return ax.figure


def plot_2d_embedding(
    Z: np.ndarray,
    y: np.ndarray,
    title: str,
    ax: Axes | None = None,
    s: float = 8,
    legend: bool = True,
    axis_labels: tuple[str, str] = ("dimension 1", "dimension 2"),
) -> Figure:
    """Scatter the first two columns of an embedding, coloured by class (``tab10``).

    Accepts ``ax`` so several embeddings (PCA, t-SNE, UMAP) can share one figure.

    Parameters
    ----------
    Z : np.ndarray
        Embedding, shape ``(n, >=2)``; only the first two columns are plotted.
    y : np.ndarray
        Integer class labels 0-9, shape ``(n,)``.
    title : str
        Axes title.
    ax : Axes or None, default None
        Axes to draw on; a new figure is created if ``None``.
    s : float, default 8
        Marker size.
    legend : bool, default True
        Draw a class legend (labelled 0-9) to the right of the axes.
    axis_labels : tuple of str, default ("dimension 1", "dimension 2")
        x- and y-axis labels.

    Returns
    -------
    Figure
        The figure containing ``ax``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for label in np.unique(y):
        mask = y == label
        ax.scatter(
            Z[mask, 0], Z[mask, 1], s=s, color=cmap(int(label) % 10), alpha=0.75,
            linewidths=0, label=str(label),
        )
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    if legend:
        ax.legend(
            title="digit", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
            markerscale=2.5, fontsize=9,
        )
    return ax.figure


def plot_components_as_images(
    components: np.ndarray,
    n: int = 8,
    image_shape: tuple[int, int] = (8, 8),
    evr: np.ndarray | None = None,
) -> Figure:
    """Show the first ``n`` principal axes as small images.

    Uses a diverging colormap centred at 0 with one symmetric scale shared by all
    panels, so red and blue mean the same pixel weight in every image.

    Parameters
    ----------
    components : np.ndarray
        Principal axes as rows, shape ``(k, prod(image_shape))``.
    n : int, default 8
        Number of components to show.
    image_shape : tuple of int, default (8, 8)
        Shape each component is reshaped to.
    evr : np.ndarray or None, default None
        Explained-variance ratios; if given, each title includes the component's EVR.

    Returns
    -------
    Figure
        Grid of images with a shared colorbar.
    """
    n = min(n, len(components))
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols + 0.8, 2.4 * nrows), squeeze=False)
    vmax = np.abs(components[:n]).max()

    for j, ax in enumerate(axes.flat):
        ax.axis("off")
        if j >= n:
            continue
        im = ax.imshow(components[j].reshape(image_shape), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        label = f"PC{j + 1}" if evr is None else f"PC{j + 1} ({evr[j]:.1%})"
        ax.set_title(label, fontsize=10)

    fig.colorbar(im, ax=axes, shrink=0.8, label="pixel weight")
    return fig


def _embedding_title(result: EmbeddingResult) -> str:
    """Method, key parameter and runtime of an embedding, for a panel title."""
    if result.name == "PCA":
        share = result.extra["explained_variance_ratio"].sum()
        detail = f"PC1+PC2 = {share:.1%} of variance"
    else:
        detail = result.key_param_label
    return f"{result.name} ({detail}), {result.runtime_s:.2f} s"


def _shared_digit_legend(fig: Figure, ax: Axes) -> None:
    """One legend for digits 0-9 beneath all panels, built from ``ax``'s handles."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, title="digit", loc="outside lower center", ncols=len(labels),
        frameon=False, markerscale=3,
    )


def plot_embeddings_side_by_side(
    results: dict[str, EmbeddingResult],
    y: np.ndarray,
    figsize: tuple[float, float] = (18, 5.5),
    s: float = 6,
) -> Figure:
    """Draw the PCA, t-SNE and UMAP embeddings next to each other (Problem 7).

    Every panel uses :func:`plot_2d_embedding` with the same marker size and
    ``tab10`` palette, and one legend is shared beneath the panels. PCA keeps its
    tick labels (its axes are PC coordinates). t-SNE and UMAP axes have no units,
    so their tick labels are hidden and a caption says so.

    Parameters
    ----------
    results : dict[str, EmbeddingResult]
        Output of :func:`src.embeddings.compute_embeddings`.
    y : np.ndarray
        Class labels, shape ``(n,)``.
    figsize : tuple of float, default (18, 5.5)
        Figure size in inches.
    s : float, default 6
        Marker size used in every panel.

    Returns
    -------
    Figure
        Figure with one panel per embedding.
    """
    fig, axes = plt.subplots(1, len(results), figsize=figsize, layout="constrained")
    for ax, (name, result) in zip(np.atleast_1d(axes), results.items()):
        if name == "PCA":
            labels = ("PC 1", "PC 2")
        else:
            labels = (f"{name} 1", f"{name} 2")
        plot_2d_embedding(
            result.Z, y, _embedding_title(result), ax=ax, s=s, legend=False, axis_labels=labels
        )
        if name != "PCA":
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.text(
                0.99, 0.01, "axes have no units", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8, color="0.4",
            )
    _shared_digit_legend(fig, np.atleast_1d(axes)[0])
    return fig


def plot_sensitivity_grid(
    grid: dict[str, list[EmbeddingResult]],
    y: np.ndarray,
    s: float = 4,
) -> Figure:
    """Show how t-SNE and UMAP change with their neighbourhood parameter.

    Row 1 holds the t-SNE runs (one per perplexity), row 2 the UMAP runs (one per
    ``n_neighbors``). Ticks are hidden, since the axes have no units, and one digit
    legend is shared beneath the grid.

    Parameters
    ----------
    grid : dict[str, list[EmbeddingResult]]
        Output of :func:`src.embeddings.sensitivity_grid`.
    y : np.ndarray
        Class labels, shape ``(n,)``.
    s : float, default 4
        Marker size used in every panel.

    Returns
    -------
    Figure
        A ``len(grid) x max(len(runs))`` grid of panels.
    """
    nrows = len(grid)
    ncols = max(len(runs) for runs in grid.values())
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.6 * nrows + 0.6), layout="constrained", squeeze=False
    )
    for row, runs in zip(axes, grid.values()):
        for ax, result in zip(row, runs):
            title = f"{result.name}, {result.key_param_label} ({result.runtime_s:.1f} s)"
            plot_2d_embedding(result.Z, y, title, ax=ax, s=s, legend=False, axis_labels=("", ""))
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in row[len(runs):]:
            ax.axis("off")
    _shared_digit_legend(fig, axes[0, 0])
    return fig


# Categorical slots for the five structure metrics (fixed order, validated for adjacent bars).
_METRIC_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")


def plot_shepard_diagrams(
    X: np.ndarray,
    results: dict[str, EmbeddingResult],
    n_pairs: int = 5000,
    random_state: int = 0,
) -> Figure:
    """Shepard diagrams: embedded distance against original distance (Problem 8).

    Each panel scatters the same random sample of point pairs. Both axes are
    divided by their maximum over the sample, so all panels share the [0, 1]
    scale. The dashed diagonal marks a perfectly distance-preserving map (up to
    scale). Titles carry Spearman's rho over *all* pairs from
    :func:`src.structure_metrics.distance_correlation`.

    Parameters
    ----------
    X : np.ndarray
        Original data, shape ``(n, p)``.
    results : dict[str, EmbeddingResult]
        Output of :func:`src.embeddings.compute_embeddings`.
    n_pairs : int, default 5000
        Number of distinct pairs ``(i, j)``, ``i != j``, to plot.
    random_state : int, default 0
        Seed for sampling the pairs.

    Returns
    -------
    Figure
        One panel per method, in the order PCA, t-SNE, UMAP.
    """
    # Imported here, not at module level: structure_metrics pulls in umap (via
    # embeddings), which the Problem 5-6 helpers in this module do not need.
    from .structure_metrics import METHOD_ORDER, distance_correlation

    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    rng = np.random.default_rng(random_state)
    i = rng.integers(0, n, size=2 * n_pairs)
    j = rng.integers(0, n, size=2 * n_pairs)
    keep = i != j
    i, j = i[keep][:n_pairs], j[keep][:n_pairs]
    d_orig = np.linalg.norm(X[i] - X[j], axis=1)
    d_orig /= d_orig.max()

    names = [m for m in METHOD_ORDER if m in results] + [m for m in results if m not in METHOD_ORDER]
    fig, axes = plt.subplots(
        1, len(names), figsize=(5.2 * len(names), 5), layout="constrained", squeeze=False
    )
    for ax, name in zip(axes[0], names):
        Z = np.asarray(results[name].Z, dtype=np.float64)
        d_emb = np.linalg.norm(Z[i] - Z[j], axis=1)
        d_emb /= d_emb.max()
        rho = distance_correlation(X, Z)
        ax.scatter(d_orig, d_emb, s=3, alpha=0.25, color="#2a78d6", linewidths=0, rasterized=True)
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=0.8)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_title(f"{name}: Spearman ρ = {rho:.3f}")
        ax.set_xlabel("original 64-d distance / max (sampled pairs)")
        ax.set_ylabel("embedded 2-d distance / max (sampled pairs)")
        ax.grid(color="0.92", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"Shepard diagrams, {len(i)} random pairs; ρ over all pairs")
    return fig


def plot_metric_bars(structure_df: pd.DataFrame) -> Figure:
    """Grouped bars of the structure metrics: methods on the x-axis, one bar per metric.

    Every metric lies in [0, 1] here, so they share one y-axis. Each bar carries
    its value, and the legend says whether the metric is local or global.

    Parameters
    ----------
    structure_df : pd.DataFrame
        Output of :func:`src.structure_metrics.structure_table` (methods as index,
        metrics as columns).

    Returns
    -------
    Figure
        The bar chart.
    """
    from .structure_metrics import METRIC_SCOPES

    metrics = [m for m in METRIC_SCOPES if m in structure_df.columns]
    methods = list(structure_df.index)
    n_metrics = len(metrics)
    width = 0.8 / n_metrics
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for m_idx, metric in enumerate(metrics):
        values = structure_df[metric].to_numpy(dtype=float)
        offsets = x - 0.4 + width * (m_idx + 0.5)
        bars = ax.bar(
            offsets, values, width=width * 0.92, color=_METRIC_COLORS[m_idx % len(_METRIC_COLORS)],
            label=f"{metric} ({METRIC_SCOPES[metric]})",
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5, color="0.25")

    ax.set_xticks(x, methods)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score (1 = perfectly preserved)")
    ax.set_title("Local vs. global structure preserved by each 2-D embedding")
    ax.grid(axis="y", color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(loc="outside right upper", frameon=False, title="metric")
    return fig
