"""Shared datasets for Problems 6-7."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits


def load_digits_features() -> tuple[np.ndarray, np.ndarray]:
    """Load the 8x8 digits dataset as a raw feature matrix and labels.

    The features are deliberately not scaled or standardised. All 64 pixels already
    share the same 0-16 intensity scale, so no feature dominates through its units.
    Several pixels (e.g. on the image border) are constant across the whole dataset,
    so standardising would divide by a zero standard deviation.

    Returns
    -------
    X : np.ndarray
        Pixel intensities, shape ``(1797, 64)``, dtype ``float64``.
    y : np.ndarray
        Digit labels 0-9, shape ``(1797,)``, integer dtype.
    """
    digits = load_digits()
    X = np.asarray(digits.data, dtype=np.float64)
    y = np.asarray(digits.target, dtype=np.int64)
    return X, y
