"""Problem 4: the two-layer MLP of Problem 2 in pure NumPy, with backpropagation and a gradient check.

Notation follows Problem 2: ``z1 = W1 x + b1``, ``h = sigma(z1)``, ``z2 = W2 h + b2``,
``y_hat = softmax(z2)``, ``W1`` of shape ``(m, d)`` and ``W2`` of shape ``(K, m)``. A mini-batch
stores samples as rows, so ``Z1 = X W1^T + b1`` is the same formula applied to every row,
and the loss is the mean cross-entropy over the batch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

PARAM_NAMES = ("W1", "b1", "W2", "b2")
GRADCHECK_TOLERANCE = 1e-5
GRADCHECK_EPS = 1e-5


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable form of 1 / (1 + e^-z)


# Elementwise activations sigma and their derivatives sigma'.
ACTIVATIONS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "relu": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0).astype(z.dtype)),
    "tanh": (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2),
    "sigmoid": (_sigmoid, lambda z: _sigmoid(z) * (1.0 - _sigmoid(z))),
}


def log_softmax(Z: np.ndarray) -> np.ndarray:
    """Row-wise ``log softmax`` via the log-sum-exp trick (subtract the row maximum first)."""
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def cross_entropy(Z2: np.ndarray, Y: np.ndarray) -> float:
    """Mean cross-entropy ``-1/B sum_i sum_k Y_ik log softmax(Z2)_ik`` of logits ``Z2``."""
    return float(-(Y * log_softmax(Z2)).sum() / Z2.shape[0])


class NumpyMLP:
    """Two-layer MLP ``softmax(W2 sigma(W1 x + b1) + b2)`` trained with cross-entropy.

    Parameters
    ----------
    n_inputs, n_hidden, n_classes : int
        Layer sizes ``d``, ``m`` and ``K``.
    activation : {"relu", "tanh", "sigmoid"}, default "relu"
        Elementwise hidden activation ``sigma``.
    random_state : int, default 0
        Seed of the weight initialisation.

    Attributes
    ----------
    params : dict of str to np.ndarray
        ``W1 (m, d)``, ``b1 (m,)``, ``W2 (K, m)``, ``b2 (K,)``, all ``float64``.

    Notes
    -----
    Weights are drawn from ``N(0, s^2 / fan_in)`` with ``s^2 = 2`` for ReLU (He) and
    ``s^2 = 1`` otherwise (LeCun), so the hidden pre-activations start at unit scale;
    biases start at zero.
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_classes: int,
        activation: str = "relu",
        random_state: int = 0,
    ) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}, got {activation!r}")
        self.activation = activation
        self._sigma, self._sigma_prime = ACTIVATIONS[activation]
        rng = np.random.default_rng(random_state)
        gain = 2.0 if activation == "relu" else 1.0
        self.params = {
            "W1": rng.normal(0.0, np.sqrt(gain / n_inputs), size=(n_hidden, n_inputs)),
            "b1": np.zeros(n_hidden),
            "W2": rng.normal(0.0, np.sqrt(1.0 / n_hidden), size=(n_classes, n_hidden)),
            "b2": np.zeros(n_classes),
        }

    def forward(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Forward pass; returns every intermediate needed by :meth:`backward`."""
        p = self.params
        Z1 = X @ p["W1"].T + p["b1"]
        H = self._sigma(Z1)
        Z2 = H @ p["W2"].T + p["b2"]
        return {"X": X, "Z1": Z1, "H": H, "Z2": Z2}

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Mean cross-entropy of the batch ``X`` with one-hot targets ``Y``."""
        return cross_entropy(self.forward(X)["Z2"], Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class labels, shape ``(n,)``."""
        return self.forward(X)["Z2"].argmax(axis=1)

    def backward(self, cache: dict[str, np.ndarray], Y: np.ndarray) -> dict[str, np.ndarray]:
        """Gradients of the mean loss, using the formulas of Problem 2.

        With ``delta2 = (y_hat - y) / B`` (Problem 1 plus the ``1/B`` of the mean) and
        ``delta1 = (delta2 W2) * sigma'(z1)``, the per-sample outer products ``delta h^T`` and
        ``delta x^T`` summed over the batch become the matrix products below.

        Parameters
        ----------
        cache : dict
            Output of :meth:`forward` on the same batch.
        Y : np.ndarray
            One-hot targets, shape ``(B, K)``.

        Returns
        -------
        dict of str to np.ndarray
            ``dL/dW1``, ``dL/db1``, ``dL/dW2``, ``dL/db2`` with the parameters' shapes.
        """
        B = Y.shape[0]
        P = np.exp(log_softmax(cache["Z2"]))
        delta2 = (P - Y) / B
        delta1 = (delta2 @ self.params["W2"]) * self._sigma_prime(cache["Z1"])
        return {
            "W2": delta2.T @ cache["H"],
            "b2": delta2.sum(axis=0),
            "W1": delta1.T @ cache["X"],
            "b1": delta1.sum(axis=0),
        }

    def loss_and_grads(self, X: np.ndarray, Y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        """Loss and analytic gradients of one batch."""
        cache = self.forward(X)
        return cross_entropy(cache["Z2"], Y), self.backward(cache, Y)


def numerical_gradient(model: NumpyMLP, X: np.ndarray, Y: np.ndarray, eps: float = GRADCHECK_EPS) -> dict[str, np.ndarray]:
    """Central finite differences ``(L(theta + eps) - L(theta - eps)) / (2 eps)`` for every entry.

    Each entry is perturbed in place and restored, so the model is unchanged afterwards.
    """
    grads = {}
    for name in PARAM_NAMES:
        theta = model.params[name]
        g = np.zeros_like(theta)
        for idx in np.ndindex(theta.shape):
            original = theta[idx]
            theta[idx] = original + eps
            loss_plus = model.loss(X, Y)
            theta[idx] = original - eps
            loss_minus = model.loss(X, Y)
            theta[idx] = original
            g[idx] = (loss_plus - loss_minus) / (2 * eps)
        grads[name] = g
    return grads


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """``||a - n|| / max(||a|| + ||n||, 1e-12)``, the usual scale-free gradient-check error."""
    return float(np.linalg.norm(analytic - numeric) / max(np.linalg.norm(analytic) + np.linalg.norm(numeric), 1e-12))


def gradient_check(
    X: np.ndarray,
    Y: np.ndarray,
    n_hidden: int,
    activations: tuple[str, ...] = ("relu", "tanh", "sigmoid"),
    eps: float = GRADCHECK_EPS,
    random_state: int = 0,
) -> pd.DataFrame:
    """Compare backprop with central differences for every parameter and activation.

    Parameters
    ----------
    X, Y : np.ndarray
        A small batch of inputs ``(B, d)`` and one-hot targets ``(B, K)``.
    n_hidden : int
        Hidden width ``m``.
    activations : tuple of str
        Activations to test; Problem 2 holds for any elementwise ``sigma``.
    eps : float, default 1e-5
        Finite-difference step.
    random_state : int, default 0
        Seed of the weight initialisation.

    Returns
    -------
    pd.DataFrame
        One row per (activation, parameter): number of entries, gradient norm, the
        relative error of the whole tensor, the worst entrywise relative error
        ``|a - n| / (|a| + |n|)`` and whether both are below ``1e-5``. Entries where both
        gradients are exactly 0 (pixels that are 0 in the whole batch, inactive ReLU units)
        agree trivially and are left out of the entrywise maximum.
    """
    rows = []
    for activation in activations:
        model = NumpyMLP(X.shape[1], n_hidden, Y.shape[1], activation=activation, random_state=random_state)
        _, analytic = model.loss_and_grads(X, Y)
        numeric = numerical_gradient(model, X, Y, eps=eps)
        for name in PARAM_NAMES:
            a, n = analytic[name], numeric[name]
            rel = relative_error(a, n)
            denom = np.abs(a) + np.abs(n)
            nonzero = denom > 0
            entrywise = float((np.abs(a - n)[nonzero] / denom[nonzero]).max()) if nonzero.any() else 0.0
            rows.append({
                "activation": activation,
                "parameter": name,
                "shape": a.shape,
                "entries": a.size,
                "grad norm": float(np.linalg.norm(a)),
                "relative error": rel,
                "max entrywise rel. error": entrywise,
                "passes": max(rel, entrywise) < GRADCHECK_TOLERANCE,
            })
    return pd.DataFrame(rows)


@dataclass
class TrainingLog:
    """Loss history of one training run.

    ``step_loss[t]`` is the mini-batch loss at step ``t`` before the update;
    ``epoch_loss[e]`` is the full training-set loss after ``e`` epochs (``e = 0`` is the
    initialisation).
    """

    step_loss: np.ndarray
    epoch_loss: np.ndarray


def train_numpy(
    model: NumpyMLP,
    X: np.ndarray,
    y: np.ndarray,
    schedule: list[list[np.ndarray]],
    lr: float,
) -> TrainingLog:
    """Train ``model`` in place with plain mini-batch SGD, ``theta <- theta - lr * grad`` (Problem 3).

    Parameters
    ----------
    model : NumpyMLP
        Network to train (modified in place).
    X, y : np.ndarray
        Training inputs ``(n, d)`` and integer labels ``(n,)``.
    schedule : list of list of np.ndarray
        Mini-batch indices per epoch, from :func:`src.data.epoch_batches`.
    lr : float
        Learning rate.

    Returns
    -------
    TrainingLog
        Per-step and per-epoch losses.
    """
    Y = np.eye(model.params["b2"].shape[0])[y]
    step_loss, epoch_loss = [], [model.loss(X, Y)]
    for batches in schedule:
        for idx in batches:
            loss, grads = model.loss_and_grads(X[idx], Y[idx])
            step_loss.append(loss)
            for name in PARAM_NAMES:
                model.params[name] -= lr * grads[name]
        epoch_loss.append(model.loss(X, Y))
    return TrainingLog(np.array(step_loss), np.array(epoch_loss))
