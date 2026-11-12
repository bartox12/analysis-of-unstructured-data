"""Problems 6-7: train the PyTorch MLP with and without L2, dropout and early stopping.

Every run logs train and validation loss and accuracy after each epoch, so the same
histories feed the learning curves (Problem 6) and the over/under-fitting diagnostics
(Problem 7).
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import Splits, epoch_batches
from .mlp_torch import DTYPE, TorchMLP

N_HIDDEN = 256
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 20


@dataclass(frozen=True)
class RegConfig:
    """One regularisation setting.

    Attributes
    ----------
    name : str
        Label used in tables and plots.
    l2 : float
        Strength ``lambda`` of the penalty ``lambda / 2 (||W1||^2 + ||W2||^2)`` added to the
        training loss (weights only, not biases).
    dropout : float
        Dropout probability on the hidden layer during training.
    early_stopping : bool
        Stop when the validation loss has not improved for ``patience`` epochs and keep the
        weights of the best epoch.
    patience : int
        Early-stopping patience in epochs.
    """

    name: str
    l2: float = 0.0
    dropout: float = 0.0
    early_stopping: bool = False
    patience: int = PATIENCE


@dataclass
class RunResult:
    """History and final scores of one training run."""

    config: RegConfig
    seed: int
    history: pd.DataFrame  # epoch, train_loss, val_loss, train_acc, val_acc
    best_epoch: int  # epoch whose weights are kept (the last one without early stopping)
    final: dict[str, float]  # train/val/test accuracy and val loss of the kept weights


@contextmanager
def _single_thread() -> Iterator[None]:
    """Run PyTorch on one CPU thread so results do not depend on the machine's core count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _evaluate(net: TorchMLP, X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Cross-entropy (data term only, no penalty) and accuracy in eval mode (dropout off)."""
    net.eval()
    with torch.no_grad():
        logits = net(X)
        return F.cross_entropy(logits, y).item(), (logits.argmax(dim=1) == y).double().mean().item()


def train_run(
    config: RegConfig,
    splits: Splits,
    seed: int = 0,
    n_hidden: int = N_HIDDEN,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
) -> RunResult:
    """Train one network with Adam (Problem 3) under ``config`` and log every epoch.

    The seed controls the weight initialisation, the dropout masks and the mini-batch
    order; the data split is fixed. The logged train loss is the plain cross-entropy on the
    whole training set in eval mode, so it is comparable across configurations (the L2
    penalty and the dropout noise only act on the optimised loss).

    Parameters
    ----------
    config : RegConfig
        Regularisation setting.
    splits : Splits
        Train / validation / test data; the test set is only scored once at the end.
    seed : int, default 0
        Random seed of this run.
    n_hidden, lr, batch_size, max_epochs : int, float, int, int
        Architecture and optimiser settings shared by all runs.

    Returns
    -------
    RunResult
        Per-epoch history, the epoch of the kept weights and the final scores.
    """
    with _single_thread():
        torch.manual_seed(seed)
        net = TorchMLP(splits.X_train.shape[1], n_hidden, int(splits.y_train.max()) + 1,
                       dropout=config.dropout)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        data = {
            part: (torch.from_numpy(getattr(splits, f"X_{part}")).to(DTYPE),
                   torch.from_numpy(getattr(splits, f"y_{part}")))
            for part in ("train", "val", "test")
        }
        X_tr, y_tr = data["train"]
        schedule = epoch_batches(len(y_tr), batch_size, max_epochs, random_state=seed)

        rows = []
        best_loss, best_epoch, best_state = np.inf, 0, copy.deepcopy(net.state_dict())
        for epoch, batches in enumerate(schedule, start=1):
            net.train()
            for idx in batches:
                i = torch.from_numpy(idx)
                loss = F.cross_entropy(net(X_tr[i]), y_tr[i])
                if config.l2 > 0:
                    loss = loss + 0.5 * config.l2 * (net.fc1.weight.square().sum() + net.fc2.weight.square().sum())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss, train_acc = _evaluate(net, *data["train"])
            val_loss, val_acc = _evaluate(net, *data["val"])
            rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "train_acc": train_acc, "val_acc": val_acc})
            if val_loss < best_loss:
                best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(net.state_dict())
            if config.early_stopping and epoch - best_epoch >= config.patience:
                break

        if config.early_stopping:
            net.load_state_dict(best_state)
        else:
            best_epoch = len(rows)
        final = {}
        for part in ("train", "val", "test"):
            final[f"{part}_loss"], final[f"{part}_acc"] = _evaluate(net, *data[part])
    return RunResult(config, seed, pd.DataFrame(rows), best_epoch, final)


def run_study(
    configs: list[RegConfig], splits: Splits, seeds: tuple[int, ...] = (0, 1, 2)
) -> dict[str, list[RunResult]]:
    """Train every configuration once per seed; returns ``{config name: [run per seed]}``."""
    return {c.name: [train_run(c, splits, seed=s) for s in seeds] for c in configs}


def strength_sweep(
    splits: Splits,
    l2_values: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
    dropout_values: tuple[float, ...] = (0.2, 0.5, 0.8),
    seeds: tuple[int, ...] = (0, 1),
) -> pd.DataFrame:
    """Train with several L2 strengths and dropout rates (no early stopping) for model selection.

    Returns
    -------
    pd.DataFrame
        One row per setting: mean over seeds of the final train / validation accuracy and
        validation loss, plus the train-validation accuracy gap. Only the validation set is
        used here; the test set is not looked at.
    """
    settings = [RegConfig("none")]
    settings += [RegConfig(f"L2 {lam:g}", l2=lam) for lam in l2_values]
    settings += [RegConfig(f"dropout {p:g}", dropout=p) for p in dropout_values]
    rows = []
    for config in settings:
        runs = [train_run(config, splits, seed=s) for s in seeds]
        rows.append({
            "regulariser": config.name.split()[0],
            "strength": config.l2 or config.dropout,
            "train acc": np.mean([r.final["train_acc"] for r in runs]),
            "val acc": np.mean([r.final["val_acc"] for r in runs]),
            "train loss": np.mean([r.final["train_loss"] for r in runs]),
            "val loss": np.mean([r.final["val_loss"] for r in runs]),
        })
    table = pd.DataFrame(rows)
    table["acc gap (train - val)"] = table["train acc"] - table["val acc"]
    return table


def best_strength(sweep: pd.DataFrame, regulariser: str) -> float:
    """Strength of ``regulariser`` (``"L2"`` or ``"dropout"``) with the lowest mean validation loss."""
    rows = sweep[sweep["regulariser"] == regulariser]
    return float(rows.loc[rows["val loss"].idxmin(), "strength"])


def default_configs(l2: float, dropout: float, patience: int = PATIENCE) -> list[RegConfig]:
    """The five settings of Problem 6: none, each regulariser alone, and all three together."""
    base = RegConfig("none")
    return [
        base,
        replace(base, name="L2", l2=l2),
        replace(base, name="dropout", dropout=dropout),
        replace(base, name="early stopping", early_stopping=True, patience=patience),
        RegConfig("L2 + dropout + early stopping", l2=l2, dropout=dropout, early_stopping=True, patience=patience),
    ]


def accuracy_table(study: dict[str, list[RunResult]]) -> pd.DataFrame:
    """Final accuracies per configuration: mean and standard deviation over seeds.

    Returns
    -------
    pd.DataFrame
        One row per configuration with train / validation / test accuracy (mean, std), the
        validation loss of the kept weights and the epoch of the kept weights.
    """
    rows = []
    for name, runs in study.items():
        row = {"configuration": name}
        for part in ("train", "val", "test"):
            accs = np.array([r.final[f"{part}_acc"] for r in runs])
            row[f"{part} acc"] = accs.mean()
            row[f"{part} acc std"] = accs.std()
        row["val loss"] = np.mean([r.final["val_loss"] for r in runs])
        row["kept epoch"] = np.mean([r.best_epoch for r in runs])
        rows.append(row)
    return pd.DataFrame(rows)


def fitting_diagnostics(study: dict[str, list[RunResult]]) -> pd.DataFrame:
    """Where each configuration under- and over-fits, read off the validation-loss curve.

    For every run we take the epoch with the lowest validation loss. Before it the model
    still improves on unseen data (the under-fitting phase); after it the validation loss
    rises while the training loss keeps falling (over-fitting). The columns are means over
    seeds.

    Returns
    -------
    pd.DataFrame
        Per configuration: epoch and value of the minimum validation loss, the validation
        loss at the last trained epoch and its rise above the minimum, the final training
        loss and the final train-validation accuracy gap.
    """
    rows = []
    for name, runs in study.items():
        stats = []
        for r in runs:
            h = r.history
            i_min = int(h["val_loss"].idxmin())
            last = h.iloc[-1]
            stats.append({
                "epochs trained": len(h),
                "min val loss epoch": int(h.loc[i_min, "epoch"]),
                "min val loss": h.loc[i_min, "val_loss"],
                "last val loss": last["val_loss"],
                "rise after min": last["val_loss"] - h.loc[i_min, "val_loss"],
                "last train loss": last["train_loss"],
                "last acc gap (train - val)": last["train_acc"] - last["val_acc"],
            })
        rows.append({"configuration": name, **pd.DataFrame(stats).mean().to_dict()})
    return pd.DataFrame(rows)
