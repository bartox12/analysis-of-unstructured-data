"""Problem 7: 2-D embeddings of the digits features with PCA, t-SNE and UMAP."""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE
from .pca import PCA

# The parameter each method's appearance depends on most; used in titles and tables.
KEY_PARAMS = {"t-SNE": "perplexity", "UMAP": "n_neighbors"}

_UMAP_WARMED_UP = False


@dataclass
class EmbeddingResult:
    """A 2-D embedding together with how it was produced.

    Attributes
    ----------
    name : str
        Method name: ``"PCA"``, ``"t-SNE"`` or ``"UMAP"``.
    Z : np.ndarray
        Embedding, shape ``(n, 2)``.
    params : dict
        Parameters passed to the method.
    runtime_s : float
        Wall-clock time of the fit/transform only, in seconds.
    extra : dict
        Method-specific diagnostics (e.g. t-SNE's ``kl_divergence_``).
    """

    name: str
    Z: np.ndarray
    params: dict[str, Any]
    runtime_s: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key_param_label(self) -> str:
        """Short ``"name = value"`` label of the method's key parameter ('' for PCA)."""
        key = KEY_PARAMS.get(self.name)
        return "" if key is None else f"{key} = {self.params[key]:g}"


def embed_pca(X: np.ndarray) -> EmbeddingResult:
    """Project ``X`` on its first two principal components (our from-scratch PCA).

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.

    Returns
    -------
    EmbeddingResult
        ``extra`` holds the explained-variance ratios of the two components.
    """
    pca = PCA(n_components=2)
    start = time.perf_counter()
    Z = pca.fit_transform(X)
    runtime = time.perf_counter() - start
    return EmbeddingResult(
        name="PCA",
        Z=Z,
        params={"n_components": 2},
        runtime_s=runtime,
        extra={"explained_variance_ratio": pca.explained_variance_ratio_.copy()},
    )


def embed_tsne(
    X: np.ndarray, perplexity: float = 30.0, random_state: int = 0, **kwargs: Any
) -> EmbeddingResult:
    """Embed ``X`` in 2-D with ``sklearn.manifold.TSNE``.

    Uses ``init="pca"`` (deterministic) and ``learning_rate="auto"``, running for
    1000 iterations.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    perplexity : float, default 30.0
        Effective number of neighbours each point attends to.
    random_state : int, default 0
        Seed for the optimisation.
    **kwargs
        Further keyword arguments for ``TSNE``.

    Returns
    -------
    EmbeddingResult
        ``extra`` holds the final ``kl_divergence_`` and ``n_iter_``.
    """
    params = {
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "max_iter": 1000,
        "random_state": random_state,
        **kwargs,
    }
    tsne = TSNE(n_components=2, **params)
    start = time.perf_counter()
    Z = tsne.fit_transform(X)
    runtime = time.perf_counter() - start
    return EmbeddingResult(
        name="t-SNE",
        Z=Z,
        params=params,
        runtime_s=runtime,
        extra={"kl_divergence": float(tsne.kl_divergence_), "n_iter": int(tsne.n_iter_)},
    )


def warmup_umap(X: np.ndarray, n: int = 200) -> None:
    """Run a tiny, untimed UMAP fit to trigger numba's JIT compilation.

    UMAP's core routines are numba functions that are compiled on first call, which
    takes several seconds. Without a warm-up, the first timed UMAP fit would mostly
    measure compilation rather than the algorithm, and it would not be comparable
    with t-SNE or with later UMAP fits in the same session.

    Parameters
    ----------
    X : np.ndarray
        Data matrix; only its first ``n`` rows are used.
    n : int, default 200
        Number of rows in the warm-up fit.
    """
    subset = X[:n]
    umap.UMAP(
        n_components=2, n_neighbors=min(15, len(subset) - 1), random_state=0, n_jobs=1
    ).fit_transform(subset)


def embed_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
    **kwargs: Any,
) -> EmbeddingResult:
    """Embed ``X`` in 2-D with ``umap.UMAP``.

    A fixed ``random_state`` makes UMAP single-threaded anyway, so ``n_jobs=1`` is
    set explicitly to avoid the "n_jobs overridden" warning. On the first call
    :func:`warmup_umap` runs once (untimed), so the reported runtime excludes numba
    compilation.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    n_neighbors : int, default 15
        Size of the local neighbourhood used to build the graph.
    min_dist : float, default 0.1
        Minimum distance between points in the embedding.
    random_state : int, default 0
        Seed for reproducibility.
    **kwargs
        Further keyword arguments for ``umap.UMAP``.

    Returns
    -------
    EmbeddingResult
        The embedding; ``extra`` is empty.
    """
    global _UMAP_WARMED_UP
    if not _UMAP_WARMED_UP:
        warmup_umap(X)
        _UMAP_WARMED_UP = True

    params = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": "euclidean",
        "random_state": random_state,
        "n_jobs": 1,
        **kwargs,
    }
    reducer = umap.UMAP(n_components=2, **params)
    start = time.perf_counter()
    Z = reducer.fit_transform(X)
    runtime = time.perf_counter() - start
    return EmbeddingResult(name="UMAP", Z=Z, params=params, runtime_s=runtime)


def compute_embeddings(
    X: np.ndarray,
    random_state: int = 0,
    tsne_kwargs: dict[str, Any] | None = None,
    umap_kwargs: dict[str, Any] | None = None,
) -> dict[str, EmbeddingResult]:
    """Compute the PCA, t-SNE and UMAP embeddings of ``X``.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    random_state : int, default 0
        Seed for t-SNE and UMAP (PCA is deterministic).
    tsne_kwargs, umap_kwargs : dict or None, default None
        Extra keyword arguments for :func:`embed_tsne` / :func:`embed_umap`.

    Returns
    -------
    dict[str, EmbeddingResult]
        Keys ``"PCA"``, ``"t-SNE"`` and ``"UMAP"``, in that order.
    """
    return {
        "PCA": embed_pca(X),
        "t-SNE": embed_tsne(X, random_state=random_state, **(tsne_kwargs or {})),
        "UMAP": embed_umap(X, random_state=random_state, **(umap_kwargs or {})),
    }


def _format_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return "[" + ", ".join(f"{v:.3f}" for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def runtime_table(results: dict[str, EmbeddingResult]) -> pd.DataFrame:
    """Tabulate the parameters, runtime and diagnostics of each embedding.

    Parameters
    ----------
    results : dict[str, EmbeddingResult]
        Output of :func:`compute_embeddings`.

    Returns
    -------
    pd.DataFrame
        Indexed by method, with columns ``params``, ``runtime_s`` and ``extra``.
    """
    rows = [
        {
            "method": name,
            "params": ", ".join(f"{k}={_format_value(v)}" for k, v in r.params.items()),
            "runtime_s": round(r.runtime_s, 3),
            "extra": ", ".join(f"{k}={_format_value(v)}" for k, v in r.extra.items()),
        }
        for name, r in results.items()
    ]
    return pd.DataFrame(rows).set_index("method")


def sensitivity_grid(
    X: np.ndarray,
    random_state: int = 0,
    perplexities: tuple[float, ...] = (5, 30, 100),
    n_neighbors: tuple[int, ...] = (5, 15, 100),
) -> dict[str, list[EmbeddingResult]]:
    """Re-run t-SNE over several perplexities and UMAP over several ``n_neighbors``.

    All other parameters stay at the defaults of :func:`embed_tsne` and
    :func:`embed_umap`.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    random_state : int, default 0
        Seed for every run.
    perplexities : tuple of float, default (5, 30, 100)
        t-SNE perplexities.
    n_neighbors : tuple of int, default (5, 15, 100)
        UMAP neighbourhood sizes.

    Returns
    -------
    dict[str, list[EmbeddingResult]]
        ``{"t-SNE": [...], "UMAP": [...]}``, one result per parameter value.
    """
    return {
        "t-SNE": [embed_tsne(X, perplexity=float(p), random_state=random_state) for p in perplexities],
        "UMAP": [embed_umap(X, n_neighbors=int(k), random_state=random_state) for k in n_neighbors],
    }
