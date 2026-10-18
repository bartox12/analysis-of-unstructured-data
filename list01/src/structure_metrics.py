"""Problem 8: how much local and global structure each 2-D embedding preserves.

All distances are Euclidean and computed in ``float64``. ``X`` is the raw 64-d
digits matrix and ``Z`` a 2-D embedding of it (rows in the same order).

Local measures (do near neighbours stay near?): :func:`trustworthiness_score`,
:func:`knn_preservation`, :func:`knn_accuracy_2d`.
Global measures (do large distances keep their meaning?): :func:`distance_correlation`,
:func:`centroid_distance_correlation`.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from .embeddings import KEY_PARAMS, EmbeddingResult, embed_pca, embed_tsne, embed_umap

METHOD_ORDER = ("PCA", "t-SNE", "UMAP")

# Column name -> (metric, "local" or "global"); used by the tables and the bar plot.
METRIC_SCOPES = {
    "trustworthiness": "local",
    "knn_preservation": "local",
    "knn_accuracy": "local",
    "distance_corr": "global",
    "centroid_corr": "global",
}


def _f64(A: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=np.float64)


def trustworthiness_score(X: np.ndarray, Z: np.ndarray, k: int = 10) -> float:
    """Trustworthiness of ``Z`` with respect to ``X`` (``sklearn.manifold.trustworthiness``).

    Penalises *false neighbours*: points among the ``k`` nearest in ``Z`` that are
    far down the neighbour ranking in ``X``. 1 means no intruders.

    Parameters
    ----------
    X : np.ndarray
        Original data, shape ``(n, p)``.
    Z : np.ndarray
        Embedding, shape ``(n, 2)``.
    k : int, default 10
        Neighbourhood size.

    Returns
    -------
    float
        Trustworthiness in ``[0, 1]``.
    """
    return float(trustworthiness(_f64(X), _f64(Z), n_neighbors=k, metric="euclidean"))


def _knn_indices(A: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's ``k`` nearest other rows, shape ``(n, k)``."""
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(_f64(A))
    # kneighbors() without a query matrix excludes each point from its own list.
    return nn.kneighbors(return_distance=False)


def knn_preservation(X: np.ndarray, Z: np.ndarray, k: int = 10) -> float:
    """Mean Jaccard overlap between each point's k-NN set in ``X`` and in ``Z``.

    The point itself is excluded from both sets. For sets of equal size ``k`` the
    Jaccard index is ``|A & B| / (2k - |A & B|)``.

    Parameters
    ----------
    X : np.ndarray
        Original data, shape ``(n, p)``.
    Z : np.ndarray
        Embedding, shape ``(n, 2)``.
    k : int, default 10
        Neighbourhood size.

    Returns
    -------
    float
        Mean Jaccard index in ``[0, 1]``; 1 means every neighbour set is kept exactly.
    """
    nx = _knn_indices(X, k)
    nz = _knn_indices(Z, k)
    shared = (nx[:, :, None] == nz[:, None, :]).sum(axis=(1, 2))
    return float(np.mean(shared / (2 * k - shared)))


def knn_accuracy_2d(
    Z: np.ndarray, y: np.ndarray, k: int = 5, cv: int = 5, random_state: int = 0
) -> float:
    """Cross-validated accuracy of a k-NN classifier trained on the embedding.

    Parameters
    ----------
    Z : np.ndarray
        Embedding, shape ``(n, 2)``.
    y : np.ndarray
        Class labels, shape ``(n,)``.
    k : int, default 5
        Number of neighbours of ``KNeighborsClassifier``.
    cv : int, default 5
        Number of stratified, shuffled folds.
    random_state : int, default 0
        Seed for the fold shuffling.

    Returns
    -------
    float
        Mean accuracy over the folds.
    """
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), _f64(Z), y, cv=folds)
    return float(scores.mean())


def distance_correlation(X: np.ndarray, Z: np.ndarray) -> float:
    """Spearman rank correlation between all pairwise distances in ``X`` and in ``Z``.

    Uses every one of the ``n (n - 1) / 2`` pairs (about 1.6M for the digits).

    Parameters
    ----------
    X : np.ndarray
        Original data, shape ``(n, p)``.
    Z : np.ndarray
        Embedding, shape ``(n, 2)``.

    Returns
    -------
    float
        Spearman's rho; 1 means the embedding keeps the ordering of all distances.
    """
    return float(spearmanr(pdist(_f64(X)), pdist(_f64(Z))).statistic)


def class_centroids(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean of the rows of ``A`` per class, in the order of ``np.unique(y)``.

    Parameters
    ----------
    A : np.ndarray
        Data, shape ``(n, d)``.
    y : np.ndarray
        Class labels, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Centroids, shape ``(n_classes, d)``.
    """
    A = _f64(A)
    return np.stack([A[y == label].mean(axis=0) for label in np.unique(y)])


def centroid_distance_correlation(X: np.ndarray, Z: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho between the pairwise class-centroid distances in ``X`` and in ``Z``.

    With 10 classes there are 45 centroid pairs. A high value means that the gaps
    *between* clusters in ``Z`` are ordered like the real gaps between classes.
    ``X`` and ``Z`` may be any two representations of the same points, e.g. two
    embeddings computed with different seeds.

    Parameters
    ----------
    X : np.ndarray
        First representation, shape ``(n, p)``.
    Z : np.ndarray
        Second representation, shape ``(n, q)``.
    y : np.ndarray
        Class labels, shape ``(n,)``.

    Returns
    -------
    float
        Spearman's rho over the centroid pairs.
    """
    dx = pdist(class_centroids(X, y))
    dz = pdist(class_centroids(Z, y))
    return float(spearmanr(dx, dz).statistic)


def structure_metrics(X: np.ndarray, Z: np.ndarray, y: np.ndarray, k: int = 10) -> dict[str, float]:
    """All five metrics for one embedding, keyed by the columns of :data:`METRIC_SCOPES`.

    Parameters
    ----------
    X : np.ndarray
        Original data, shape ``(n, p)``.
    Z : np.ndarray
        Embedding, shape ``(n, 2)``.
    y : np.ndarray
        Class labels, shape ``(n,)``.
    k : int, default 10
        Neighbourhood size for trustworthiness and kNN preservation. The kNN
        classifier keeps its own default (``k = 5``, 5 folds).

    Returns
    -------
    dict[str, float]
        ``trustworthiness``, ``knn_preservation``, ``knn_accuracy``,
        ``distance_corr`` and ``centroid_corr``.
    """
    return {
        "trustworthiness": trustworthiness_score(X, Z, k=k),
        "knn_preservation": knn_preservation(X, Z, k=k),
        "knn_accuracy": knn_accuracy_2d(Z, y),
        "distance_corr": distance_correlation(X, Z),
        "centroid_corr": centroid_distance_correlation(X, Z, y),
    }


def structure_table(
    X: np.ndarray, results: dict[str, EmbeddingResult], y: np.ndarray, k: int = 10
) -> pd.DataFrame:
    """Local and global structure metrics for each embedding of Problem 7.

    Parameters
    ----------
    X : np.ndarray
        Original data, shape ``(n, p)``.
    results : dict[str, EmbeddingResult]
        Output of :func:`src.embeddings.compute_embeddings`.
    y : np.ndarray
        Class labels, shape ``(n,)``.
    k : int, default 10
        Neighbourhood size, see :func:`structure_metrics`.

    Returns
    -------
    pd.DataFrame
        Indexed by method (PCA, t-SNE, UMAP first, then any others), one column
        per metric.
    """
    names = [m for m in METHOD_ORDER if m in results] + [m for m in results if m not in METHOD_ORDER]
    rows = [{"method": m, **structure_metrics(X, results[m].Z, y, k=k)} for m in names]
    return pd.DataFrame(rows).set_index("method")


def sensitivity_table(
    X: np.ndarray, grid: dict[str, list[EmbeddingResult]], y: np.ndarray, k: int = 10
) -> pd.DataFrame:
    """The structure metrics for every run of the Problem 7 sensitivity grid.

    Parameters
    ----------
    X : np.ndarray
        Original data, shape ``(n, p)``.
    grid : dict[str, list[EmbeddingResult]]
        Output of :func:`src.embeddings.sensitivity_grid`.
    y : np.ndarray
        Class labels, shape ``(n,)``.
    k : int, default 10
        Neighbourhood size, see :func:`structure_metrics`.

    Returns
    -------
    pd.DataFrame
        One row per run, with columns ``method``, ``parameter``, ``value`` and the
        five metrics.
    """
    rows = []
    for method, runs in grid.items():
        for r in runs:
            param = KEY_PARAMS[r.name]
            rows.append(
                {"method": method, "parameter": param, "value": r.params[param],
                 **structure_metrics(X, r.Z, y, k=k)}
            )
    return pd.DataFrame(rows)


def seed_stability(
    X: np.ndarray,
    y: np.ndarray,
    seeds: tuple[int, ...] = (0, 1, 2),
    k: int = 10,
    methods: tuple[str, ...] = METHOD_ORDER,
    results: dict[str, EmbeddingResult] | None = None,
    tsne_kwargs: dict[str, Any] | None = None,
    umap_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Do the gaps between clusters survive a change of random seed?

    Re-runs t-SNE and UMAP once per seed (default parameters otherwise) and, for
    every pair of seeds, compares the two embeddings *with each other* through
    :func:`centroid_distance_correlation` on their 45 class-centroid distances.
    Each run's trustworthiness is reported alongside. PCA is deterministic, so it
    gets a single row with rho = 1.

    Note that with the default ``init="pca"`` t-SNE is deterministic too: the seed
    only enters through the initialisation, so all seeds give the same embedding.
    Pass ``tsne_kwargs={"init": "random"}`` to see the effect of the seed.

    Parameters
    ----------
    X : np.ndarray
        Original data, shape ``(n, p)``.
    y : np.ndarray
        Class labels, shape ``(n,)``.
    seeds : tuple of int, default (0, 1, 2)
        Seeds to compare.
    k : int, default 10
        Neighbourhood size for trustworthiness.
    methods : tuple of str, default ("PCA", "t-SNE", "UMAP")
        Which methods to include.
    results : dict[str, EmbeddingResult] or None, default None
        Embeddings already computed with default parameters (e.g. from
        :func:`src.embeddings.compute_embeddings`). A t-SNE/UMAP result is reused
        for the seed it was computed with instead of being refitted, provided no
        extra kwargs are given for that method.
    tsne_kwargs, umap_kwargs : dict or None, default None
        Extra keyword arguments for :func:`src.embeddings.embed_tsne` /
        :func:`src.embeddings.embed_umap`.

    Returns
    -------
    pd.DataFrame
        Columns ``method`` (with any non-default kwargs appended), ``seed_a``, ``seed_b``, ``centroid_corr_between_seeds``,
        ``trust_seed_a`` and ``trust_seed_b``.
    """
    results = results or {}
    embedders = {
        "t-SNE": (embed_tsne, tsne_kwargs or {}),
        "UMAP": (embed_umap, umap_kwargs or {}),
    }

    rows: list[dict[str, Any]] = []
    if "PCA" in methods:
        pca = results.get("PCA") or embed_pca(X)
        pca_trust = trustworthiness_score(X, pca.Z, k=k)
        rows.append(
            {"method": "PCA", "seed_a": "-", "seed_b": "-", "centroid_corr_between_seeds": 1.0,
             "trust_seed_a": pca_trust, "trust_seed_b": pca_trust}
        )
    for method, (embed, kwargs) in embedders.items():
        if method not in methods:
            continue
        label = method
        if kwargs:
            label += " (" + ", ".join(f"{key}={val}" for key, val in kwargs.items()) + ")"
        cached = results.get(method)
        Zs: dict[int, np.ndarray] = {}
        for seed in seeds:
            if cached is not None and not kwargs and cached.params.get("random_state") == seed:
                Zs[seed] = cached.Z
            else:
                Zs[seed] = embed(X, random_state=seed, **kwargs).Z
        trust = {seed: trustworthiness_score(X, Z, k=k) for seed, Z in Zs.items()}
        for a, b in combinations(seeds, 2):
            rows.append(
                {"method": label, "seed_a": a, "seed_b": b,
                 "centroid_corr_between_seeds": centroid_distance_correlation(Zs[a], Zs[b], y),
                 "trust_seed_a": trust[a], "trust_seed_b": trust[b]}
            )
    return pd.DataFrame(rows)


def relative_contrast(
    X: np.ndarray, n_queries: int = 200, random_state: int = 0
) -> dict[str, float]:
    """Empirical relative contrast ``(D_max - D_min) / D_min`` of a point cloud.

    Query points ``q`` are drawn (without replacement) from the rows of ``X``.
    ``D_min`` and ``D_max`` are the distances from ``q`` to its nearest and
    farthest *other* row. This is the quantity of Problem 3, which tends to 0 as
    the dimension grows for i.i.d. data.

    Parameters
    ----------
    X : np.ndarray
        Points, shape ``(n, d)``; rows must be distinct, so that ``D_min > 0``.
    n_queries : int, default 200
        Number of query points (capped at ``n``).
    random_state : int, default 0
        Seed for choosing the queries.

    Returns
    -------
    dict[str, float]
        ``mean`` and ``std`` of the relative contrast over the queries, plus ``n``,
        ``d`` and ``n_queries``.
    """
    X = _f64(X)
    n, d = X.shape
    rng = np.random.default_rng(random_state)
    queries = rng.choice(n, size=min(n_queries, n), replace=False)
    D = cdist(X[queries], X)
    D[np.arange(len(queries)), queries] = np.nan  # drop the distance of q to itself
    d_min, d_max = np.nanmin(D, axis=1), np.nanmax(D, axis=1)
    contrast = (d_max - d_min) / d_min
    return {"mean": float(contrast.mean()), "std": float(contrast.std()),
            "n": n, "d": d, "n_queries": len(queries)}


def relative_contrast_uniform(
    n: int, d: int, n_queries: int = 200, random_state: int = 0
) -> dict[str, float]:
    """:func:`relative_contrast` of ``n`` i.i.d. uniform points in ``[0, 1]^d``.

    Parameters
    ----------
    n : int
        Number of points.
    d : int
        Dimension.
    n_queries : int, default 200
        Number of query points.
    random_state : int, default 0
        Seed for the points and for choosing the queries.

    Returns
    -------
    dict[str, float]
        As in :func:`relative_contrast`.
    """
    U = np.random.default_rng(random_state).uniform(size=(n, d))
    return relative_contrast(U, n_queries=n_queries, random_state=random_state)


def relative_contrast_table(
    X: np.ndarray, n_queries: int = 200, random_state: int = 0
) -> pd.DataFrame:
    """Relative contrast of the digits next to uniform clouds of the same size.

    Parameters
    ----------
    X : np.ndarray
        Digits features, shape ``(n, 64)``.
    n_queries : int, default 200
        Number of query points per cloud.
    random_state : int, default 0
        Seed for the uniform points and the queries.

    Returns
    -------
    pd.DataFrame
        Rows "digits (64-d)", "uniform [0,1]^64" and "uniform [0,1]^2", with
        columns ``n``, ``d``, ``n_queries``, ``mean`` and ``std``.
    """
    n, d = X.shape
    rows = {
        f"digits ({d}-d)": relative_contrast(X, n_queries, random_state),
        f"uniform [0,1]^{d}": relative_contrast_uniform(n, d, n_queries, random_state),
        "uniform [0,1]^2": relative_contrast_uniform(n, 2, n_queries, random_state),
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "point cloud"
    return df[["n", "d", "n_queries", "mean", "std"]]
