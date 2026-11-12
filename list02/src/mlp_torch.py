"""Problem 5: the same two-layer MLP in PyTorch with autograd, checked against the NumPy version.

Everything runs in ``float64`` so that the only differences left between the two
implementations are floating-point rounding in a different order of operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .mlp_numpy import PARAM_NAMES, NumpyMLP, TrainingLog

DTYPE = torch.float64
AGREEMENT_TOLERANCE = 1e-10

_TORCH_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class TorchMLP(nn.Module):
    """``softmax(W2 sigma(W1 x + b1) + b2)`` as an ``nn.Module`` (the softmax lives in the loss).

    Parameters
    ----------
    n_inputs, n_hidden, n_classes : int
        Layer sizes ``d``, ``m`` and ``K``.
    activation : {"relu", "tanh", "sigmoid"}, default "relu"
        Hidden activation ``sigma``.
    dropout : float, default 0.0
        Dropout probability on the hidden layer ``h`` (used in Problem 6; 0 disables it).
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_classes: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden, dtype=DTYPE)
        self.act = _TORCH_ACTIVATIONS[activation]()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden, n_classes, dtype=DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logits ``z2``."""
        return self.fc2(self.drop(self.act(self.fc1(x))))

    @classmethod
    def from_numpy(cls, model: NumpyMLP, dropout: float = 0.0) -> TorchMLP:
        """Build a network with exactly the weights of a :class:`NumpyMLP`.

        ``nn.Linear`` stores its weight as ``(out_features, in_features)``, which is the
        shape of ``W1`` and ``W2`` in Problem 2, so the arrays are copied as they are.
        """
        W1 = model.params["W1"]
        net = cls(W1.shape[1], W1.shape[0], model.params["b2"].shape[0], model.activation, dropout)
        with torch.no_grad():
            net.fc1.weight.copy_(torch.from_numpy(model.params["W1"]))
            net.fc1.bias.copy_(torch.from_numpy(model.params["b1"]))
            net.fc2.weight.copy_(torch.from_numpy(model.params["W2"]))
            net.fc2.bias.copy_(torch.from_numpy(model.params["b2"]))
        return net

    def params_by_name(self) -> dict[str, torch.Tensor]:
        """Parameters under the names of Problem 2."""
        return {"W1": self.fc1.weight, "b1": self.fc1.bias, "W2": self.fc2.weight, "b2": self.fc2.bias}


def train_torch(
    net: TorchMLP,
    X: np.ndarray,
    y: np.ndarray,
    schedule: list[list[np.ndarray]],
    lr: float,
) -> TrainingLog:
    """Train ``net`` in place with ``torch.optim.SGD`` and autograd, on the given batch schedule.

    Mirrors :func:`src.mlp_numpy.train_numpy`: same mini-batches in the same order, same
    learning rate, losses logged at the same moments.
    """
    Xt, yt = torch.from_numpy(X).to(DTYPE), torch.from_numpy(y)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    def full_loss() -> float:
        net.eval()
        with torch.no_grad():
            return F.cross_entropy(net(Xt), yt).item()

    step_loss, epoch_loss = [], [full_loss()]
    for batches in schedule:
        net.train()
        for idx in batches:
            i = torch.from_numpy(idx)
            loss = F.cross_entropy(net(Xt[i]), yt[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_loss.append(loss.item())
        epoch_loss.append(full_loss())
    return TrainingLog(np.array(step_loss), np.array(epoch_loss))


def compare_gradients(model: NumpyMLP, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Autograd gradients versus our manual backward pass on one batch, at the same weights.

    Returns
    -------
    pd.DataFrame
        One row per parameter with the largest absolute difference and whether it is below
        ``1e-10``.
    """
    Y = np.eye(model.params["b2"].shape[0])[y]
    _, ours = model.loss_and_grads(X, Y)
    net = TorchMLP.from_numpy(model)
    F.cross_entropy(net(torch.from_numpy(X)), torch.from_numpy(y)).backward()
    rows = []
    for name, p in net.params_by_name().items():
        diff = float(np.abs(p.grad.numpy() - ours[name]).max())
        rows.append({"parameter": name, "shape": tuple(p.shape), "max |autograd - numpy|": diff,
                     "passes": diff < AGREEMENT_TOLERANCE})
    return pd.DataFrame(rows)


def compare_logs(log_np: TrainingLog, log_torch: TrainingLog) -> pd.DataFrame:
    """Summarise how far the NumPy and PyTorch loss curves are apart.

    Returns
    -------
    pd.DataFrame
        One row per curve (per step, per epoch) with the number of points, the largest
        absolute and relative difference, the final losses and a pass flag (``< 1e-10``).
    """
    rows = []
    for curve in ("step_loss", "epoch_loss"):
        a, b = getattr(log_np, curve), getattr(log_torch, curve)
        diff = np.abs(a - b)
        rows.append({
            "curve": curve.replace("_", " "),
            "points": a.size,
            "max |numpy - torch|": float(diff.max()),
            "max relative diff": float((diff / np.abs(a)).max()),
            "final loss numpy": float(a[-1]),
            "final loss torch": float(b[-1]),
            "passes": float(diff.max()) < AGREEMENT_TOLERANCE,
        })
    return pd.DataFrame(rows)


def final_weight_difference(model: NumpyMLP, net: TorchMLP) -> float:
    """Largest absolute difference between the trained NumPy and PyTorch weights."""
    torch_params = net.params_by_name()
    return max(float(np.abs(model.params[n] - torch_params[n].detach().numpy()).max()) for n in PARAM_NAMES)
