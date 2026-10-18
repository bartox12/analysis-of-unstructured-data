"""Problem 6: PCA from scratch via the SVD, plus the comparison with sklearn.decomposition.PCA.

The PCA itself uses NumPy only. sklearn is imported locally inside
:func:`compare_with_sklearn`, which exists solely to check our implementation against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOLERANCE = 1e-6


def _flip_signs(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make the SVD signs deterministic.

    Singular vectors are only defined up to sign: ``(u_j, v_j)`` and ``(-u_j, -v_j)``
    give the same product ``U S Vt``. For each component ``j`` we flip both so that the
    entry of ``Vt[j]`` with the largest absolute value is positive.

    Parameters
    ----------
    U : np.ndarray
        Left singular vectors, shape ``(n, r)``.
    Vt : np.ndarray
        Right singular vectors as rows, shape ``(r, p)``.

    Returns
    -------
    U, Vt : tuple of np.ndarray
        Sign-corrected copies with the same shapes.
    """
    max_idx = np.argmax(np.abs(Vt), axis=1)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), max_idx])
    signs[signs == 0] = 1.0
    return U * signs, Vt * signs[:, None]


class PCA:
    """Principal component analysis via the SVD of the centred data matrix.

    With ``Xc = X - mean = U S Vt``, the principal directions are the rows of ``Vt``,
    the scores are ``Xc Vt.T = U S``, and the variance along direction ``j`` is
    ``S_j**2 / (n - 1)`` (sklearn's convention; dividing by ``n`` instead changes
    neither the directions nor the ratios).

    Parameters
    ----------
    n_components : int or None, default None
        Number of components to keep; ``None`` keeps ``min(n_samples, n_features)``.

    Attributes
    ----------
    mean_ : np.ndarray
        Per-feature mean, shape ``(p,)``.
    components_ : np.ndarray
        Principal directions as rows, shape ``(k, p)``.
    singular_values_ : np.ndarray
        Top ``k`` singular values of the centred data, shape ``(k,)``.
    explained_variance_ : np.ndarray
        ``singular_values_**2 / (n - 1)``, shape ``(k,)``.
    explained_variance_ratio_ : np.ndarray
        ``sigma_j**2 / sum_l sigma_l**2``, where the sum runs over *all* singular
        values, not only the kept ones; shape ``(k,)``.
    n_components_ : int
        Number of kept components ``k``.
    n_samples_ : int
        Number of training samples ``n``.
    """

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> PCA:
        """Fit the model on ``X``.

        Parameters
        ----------
        X : np.ndarray
            Data matrix, shape ``(n, p)``.

        Returns
        -------
        PCA
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` is not 2-D or ``n_components`` is not in ``[1, min(n, p)]``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        n, p = X.shape
        max_k = min(n, p)
        k = max_k if self.n_components is None else self.n_components
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= max_k:
            raise ValueError(
                f"n_components must be an int in [1, {max_k}] (or None), got {self.n_components!r}."
            )

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        U, Vt = _flip_signs(U, Vt)

        self._all_singular_values = S
        self.n_samples_ = n
        self.n_components_ = int(k)
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        self.explained_variance_ = S[:k] ** 2 / (n - 1)
        self.explained_variance_ratio_ = S[:k] ** 2 / np.sum(S**2)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError("This PCA instance is not fitted yet; call fit(X) first.")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project ``X`` onto the principal directions.

        Parameters
        ----------
        X : np.ndarray
            Data matrix, shape ``(m, p)``.

        Returns
        -------
        np.ndarray
            Scores ``(X - mean_) @ components_.T``, shape ``(m, k)``.
        """
        self._check_fitted()
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on ``X`` and return its scores, shape ``(n, k)``."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map scores back to feature space.

        Parameters
        ----------
        Z : np.ndarray
            Scores, shape ``(m, k)``.

        Returns
        -------
        np.ndarray
            Reconstruction ``Z @ components_ + mean_``, shape ``(m, p)``.
        """
        self._check_fitted()
        return np.asarray(Z, dtype=np.float64) @ self.components_ + self.mean_


def n_components_for_variance(evr: np.ndarray, threshold: float) -> int:
    """Smallest number of components whose cumulative explained variance reaches ``threshold``.

    Parameters
    ----------
    evr : np.ndarray
        Explained-variance ratios in decreasing order, shape ``(k,)``.
    threshold : float
        Target fraction of variance in ``(0, 1]``.

    Returns
    -------
    int
        The smallest ``k`` (1-based) with ``sum(evr[:k]) >= threshold``.

    Raises
    ------
    ValueError
        If ``threshold`` is outside ``(0, 1]`` or the ratios never reach it.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}.")
    cumulative = np.cumsum(evr)
    k = int(np.searchsorted(cumulative, threshold, side="left")) + 1
    if k > len(cumulative):
        raise ValueError(
            f"The given ratios only reach {cumulative[-1]:.6f} < threshold {threshold}."
        )
    return k


def theory_checks(X: np.ndarray, ks: tuple[int, ...] = (2, 5, 10, 20)) -> pd.DataFrame:
    """Confirm the Problem 4 identities numerically with our :class:`PCA`.

    (a) ``EVR_j = sigma_j**2 / sum_l sigma_l**2``. The denominator is computed
        directly as ``||Xc||_F**2`` (the total squared deviation of the data), so the
        check does not just repeat the formula used inside ``fit``.
    (b) The eigenvalues of ``np.cov(X, rowvar=False)`` (``np.linalg.eigh``, sorted
        in descending order) equal ``explained_variance_ = sigma_j**2 / (n - 1)``.
    (c) For each ``k`` in ``ks``, the reconstruction error
        ``sum_i ||x_i - xhat_i||**2`` with ``k`` components equals
        ``sum_{j>k} sigma_j**2``.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    ks : tuple of int, default (2, 5, 10, 20)
        Numbers of components for check (c).

    Returns
    -------
    pd.DataFrame
        One row per check with the PCA-side value, the theoretical value, the
        absolute error and the relative error (``abs_error / |theory|``). For the
        vector checks (a) and (b) the values are the maxima over components and
        the errors are max-abs and max-abs relative to the largest entry.
    """
    X = np.asarray(X, dtype=np.float64)
    full = PCA().fit(X)
    Xc = X - full.mean_
    sv2 = full._all_singular_values**2

    rows = []

    evr_theory = sv2 / np.sum(Xc**2)
    err = np.abs(full.explained_variance_ratio_ - evr_theory).max()
    rows.append(
        ("(a) EVR = sigma^2 / sum sigma^2", None, full.explained_variance_ratio_.max(),
         evr_theory.max(), err, err / np.abs(evr_theory).max())
    )

    eigvals = np.linalg.eigh(np.cov(X, rowvar=False))[0][::-1]
    err = np.abs(full.explained_variance_ - eigvals).max()
    rows.append(
        ("(b) eig(cov) = explained_variance_", None, full.explained_variance_.max(),
         eigvals.max(), err, err / np.abs(eigvals).max())
    )

    for k in ks:
        pca_k = PCA(k).fit(X)
        recon_error = np.sum((X - pca_k.inverse_transform(pca_k.transform(X))) ** 2)
        tail = np.sum(sv2[k:])
        err = abs(recon_error - tail)
        rows.append(
            ("(c) recon. error = sum_{j>k} sigma_j^2", k, recon_error, tail, err, err / tail)
        )

    table = pd.DataFrame(
        rows, columns=["check", "k", "pca_value", "theory_value", "abs_error", "rel_error"]
    )
    table["k"] = table["k"].astype("Int64")
    return table


def compare_with_sklearn(
    X: np.ndarray,
    n_components: int | None = None,
    n_check_components: int = 10,
) -> pd.DataFrame:
    """Compare our :class:`PCA` with ``sklearn.decomposition.PCA(svd_solver="full")``.

    Spectral quantities (ratios, variances, singular values) are compared for all
    kept components. Directions and scores are compared only for the first
    ``n_check_components``: on the digits data several pixels are constant, so the
    trailing singular values are ~0 and their directions are not uniquely defined
    (any orthonormal basis of that near-null space is a valid answer). Each sklearn
    component is sign-aligned to ours (multiplied by the sign of their dot product)
    before comparing, since singular vectors are only defined up to sign. The
    reconstruction ``inverse_transform(transform(X))`` with ``k = 10`` is sign-invariant.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    n_components : int or None, default None
        Passed to both implementations.
    n_check_components : int, default 10
        Number of leading components whose directions and scores are compared.

    Returns
    -------
    pd.DataFrame
        One row per quantity with ``max_abs_diff`` and ``passes``
        (``max_abs_diff < 1e-6``), indexed by quantity.
    """
    from sklearn.decomposition import PCA as SklearnPCA

    ours = PCA(n_components).fit(X)
    theirs = SklearnPCA(n_components=n_components, svd_solver="full").fit(X)

    m = n_check_components
    signs = np.sign(np.sum(ours.components_[:m] * theirs.components_[:m], axis=1))
    their_components = theirs.components_[:m] * signs[:, None]
    their_scores = theirs.transform(X)[:, :m] * signs

    k_rec = 10
    ours_rec = PCA(k_rec).fit(X)
    theirs_rec = SklearnPCA(n_components=k_rec, svd_solver="full").fit(X)

    diffs = {
        "explained_variance_ratio_": np.abs(
            ours.explained_variance_ratio_ - theirs.explained_variance_ratio_
        ).max(),
        "explained_variance_": np.abs(ours.explained_variance_ - theirs.explained_variance_).max(),
        "singular_values_": np.abs(ours.singular_values_ - theirs.singular_values_).max(),
        f"components_[:{m}]": np.abs(ours.components_[:m] - their_components).max(),
        f"transform(X)[:, :{m}]": np.abs(ours.transform(X)[:, :m] - their_scores).max(),
        f"inverse_transform(transform(X)), k={k_rec}": np.abs(
            ours_rec.inverse_transform(ours_rec.transform(X))
            - theirs_rec.inverse_transform(theirs_rec.transform(X))
        ).max(),
    }
    table = pd.DataFrame({"max_abs_diff": pd.Series(diffs, dtype=float)})
    table.index.name = "quantity"
    table["passes"] = table["max_abs_diff"] < TOLERANCE
    return table
