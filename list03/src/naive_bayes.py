"""Multinomial Naive Bayes from scratch (Problems 2 and 4), checked against scikit-learn."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logsumexp
from sklearn.naive_bayes import MultinomialNB

TOLERANCE = 1e-10


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes with Laplace/Lidstone smoothing (Problem 2).

    Fitting is counting: ``theta_{t|c} = (n_{t,c} + alpha) / (sum_t' n_{t',c} + alpha |V|)``
    and ``pi_c = N_c / N``. Prediction is the linear rule
    ``argmax_c  log pi_c + sum_t x_t log theta_{t|c}``.

    Parameters
    ----------
    alpha : float, default 1.0
        Pseudo-count added to every term of every class (1 = Laplace).

    Attributes
    ----------
    classes_ : np.ndarray
        Sorted class labels.
    class_count_ : np.ndarray
        Number of training documents per class, shape ``(K,)``.
    feature_count_ : np.ndarray
        ``n_{t,c}``: summed feature values per class, shape ``(K, |V|)``.
    class_log_prior_ : np.ndarray
        ``log pi_c``, shape ``(K,)``; the intercepts ``b_c`` of the linear rule.
    feature_log_prob_ : np.ndarray
        ``log theta_{t|c}``, shape ``(K, |V|)``; the weights ``w_c`` of the linear rule.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: sp.spmatrix | np.ndarray, y: np.ndarray) -> MultinomialNaiveBayes:
        """Estimate the class priors and smoothed term probabilities by counting.

        Parameters
        ----------
        X : sparse matrix or np.ndarray
            Non-negative features (counts or TF-IDF weights), shape ``(n, |V|)``.
        y : np.ndarray
            Labels, shape ``(n,)``.

        Returns
        -------
        MultinomialNaiveBayes
            ``self``.
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        onehot = (y[:, None] == self.classes_[None, :]).astype(np.float64)  # (n, K)
        self.class_count_ = onehot.sum(axis=0)
        self.feature_count_ = np.asarray((sp.csr_matrix(X).T @ onehot).T)  # (K, |V|)

        smoothed = self.feature_count_ + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log(self.class_count_) - np.log(self.class_count_.sum())
        return self

    def decision_function(self, X: sp.spmatrix | np.ndarray) -> np.ndarray:
        """Joint log-likelihood ``log pi_c + x . log theta_c`` up to a class-independent constant.

        Parameters
        ----------
        X : sparse matrix or np.ndarray
            Features, shape ``(n, |V|)``.

        Returns
        -------
        np.ndarray
            Scores ``X W^T + b``, shape ``(n, K)``.
        """
        return np.asarray(X @ self.feature_log_prob_.T) + self.class_log_prior_

    def predict_proba(self, X: sp.spmatrix | np.ndarray) -> np.ndarray:
        """Posterior class probabilities: the softmax of the linear scores."""
        scores = self.decision_function(X)
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))

    def predict(self, X: sp.spmatrix | np.ndarray) -> np.ndarray:
        """Class with the largest linear score."""
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


def compare_with_sklearn(
    X_train: sp.spmatrix, y_train: np.ndarray, X_test: sp.spmatrix, alpha: float
) -> pd.DataFrame:
    """Fit our Naive Bayes and ``sklearn.naive_bayes.MultinomialNB`` and compare them.

    Parameters
    ----------
    X_train, X_test : sparse matrix
        Training and test features.
    y_train : np.ndarray
        Training labels.
    alpha : float
        Smoothing pseudo-count used by both models.

    Returns
    -------
    pd.DataFrame
        One row per quantity with the maximum absolute difference and ``passes``.
    """
    ours = MultinomialNaiveBayes(alpha=alpha).fit(X_train, y_train)
    ref = MultinomialNB(alpha=alpha).fit(X_train, y_train)
    rows = [
        ("class_log_prior_ (b_c)", np.max(np.abs(ours.class_log_prior_ - ref.class_log_prior_))),
        ("feature_log_prob_ (w_c)", np.max(np.abs(ours.feature_log_prob_ - ref.feature_log_prob_))),
        ("X W^T + b vs. predict_joint_log_proba", np.max(np.abs(ours.decision_function(X_test) - ref.predict_joint_log_proba(X_test)))),
        ("predict_proba", np.max(np.abs(ours.predict_proba(X_test) - ref.predict_proba(X_test)))),
        ("share of different predictions", np.mean(ours.predict(X_test) != ref.predict(X_test))),
    ]
    table = pd.DataFrame(rows, columns=["quantity", "max_abs_diff"]).set_index("quantity")
    table["passes"] = table["max_abs_diff"] < TOLERANCE
    return table
