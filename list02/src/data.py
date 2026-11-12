"""Shared data for Problems 4-7: the Problem-Set-1 digits features, their splits and mini-batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

PIXEL_MAX = 16.0
N_CLASSES = 10


def load_digits_features() -> tuple[np.ndarray, np.ndarray]:
    """Load the 8x8 digits as in Problem Set 1, with the pixels scaled to [0, 1].

    The 64 pixels share one 0-16 intensity scale, so dividing by 16 keeps their relative
    sizes (the same argument as list01 for not standardising) while giving the network
    inputs of order 1. Standardising per pixel would divide by a zero standard deviation
    for the constant border pixels.

    Returns
    -------
    X : np.ndarray
        Pixel intensities in [0, 1], shape ``(1797, 64)``, dtype ``float64``.
    y : np.ndarray
        Digit labels 0-9, shape ``(1797,)``, dtype ``int64``.
    """
    digits = load_digits()
    X = np.asarray(digits.data, dtype=np.float64) / PIXEL_MAX
    y = np.asarray(digits.target, dtype=np.int64)
    return X, y


@dataclass(frozen=True)
class Splits:
    """Stratified train / validation / test split of ``(X, y)``."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 0,
) -> Splits:
    """Split into train / validation / test sets, stratified by class.

    The validation set drives early stopping and every model choice; the test set is
    only used once, for the final table.

    Parameters
    ----------
    X, y : np.ndarray
        Features ``(n, d)`` and labels ``(n,)``.
    val_size, test_size : float
        Fractions of the full data set used for validation and test.
    random_state : int, default 0
        Seed of the split.

    Returns
    -------
    Splits
        The six arrays.
    """
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size / (1 - test_size), stratify=y_rest,
        random_state=random_state,
    )
    return Splits(X_train, y_train, X_val, y_val, X_test, y_test)


def one_hot(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """One-hot encode integer labels, shape ``(n, n_classes)``, dtype ``float64``."""
    Y = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def epoch_batches(
    n: int, batch_size: int, n_epochs: int, random_state: int = 0
) -> list[list[np.ndarray]]:
    """Pre-compute the shuffled mini-batch indices of every epoch.

    Both the NumPy and the PyTorch network consume the same list, so they see exactly
    the same samples in the same order (Problem 5).

    Parameters
    ----------
    n : int
        Number of training samples.
    batch_size : int
        Mini-batch size; the last batch of an epoch may be smaller.
    n_epochs : int
        Number of epochs.
    random_state : int, default 0
        Seed of the shuffling.

    Returns
    -------
    list of list of np.ndarray
        ``schedule[epoch][step]`` holds the sample indices of one mini-batch.
    """
    rng = np.random.default_rng(random_state)
    schedule = []
    for _ in range(n_epochs):
        order = rng.permutation(n)
        schedule.append([order[i : i + batch_size] for i in range(0, n, batch_size)])
    return schedule
