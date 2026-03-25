"""PyTorch MLP classifiers wrapped in an sklearn-compatible interface."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

log = logging.getLogger("classify")


# ---------------------------------------------------------------------------
# MLP architectures
# ---------------------------------------------------------------------------

def _build_mlp(input_dim: int, n_classes: int, hidden_layers: list[int], dropout: float = 0.3) -> nn.Sequential:
    """Build a generic MLP: [Linear → BN → ReLU → Dropout] × N → Linear."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_layers:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


MLP_CONFIGS: dict[str, list[int]] = {
    "MLP_Small":  [128],
    "MLP_Medium": [256, 128],
    "MLP_Large":  [512, 256, 128],
}


# ---------------------------------------------------------------------------
# Sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class MLPClassifier:
    """Thin wrapper so MLP can be used like an sklearn estimator (fit/predict)."""

    def __init__(
        self,
        name: str,
        hidden_layers: list[int],
        *,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        dropout: float = 0.3,
        patience: int = 8,
        device: str | None = None,
    ):
        self.name = name
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Sequential | None = None
        self.n_classes: int = 0

    # -- sklearn interface --------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        self.n_classes = int(y.max()) + 1
        self.model = _build_mlp(X.shape[1], self.n_classes, self.hidden_layers, self.dropout).to(self.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        epoch_bar = tqdm(range(1, self.epochs + 1), desc=f"  {self.name}", unit="epoch", leave=False)
        for epoch in epoch_bar:
            # --- train ---
            self.model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimiser.step()
                running_loss += loss.item() * len(xb)
            train_loss = running_loss / len(train_ds)

            # --- val ---
            if has_val:
                val_loss = self._compute_loss(X_val, y_val, criterion)
                epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        log.info("%s early-stopped at epoch %d (val_loss=%.4f)", self.name, epoch, best_val_loss)
                        break
            else:
                epoch_bar.set_postfix(train=f"{train_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(xt)
        return logits.argmax(dim=1).cpu().numpy()

    # -- helpers ------------------------------------------------------------

    def _compute_loss(self, X, y, criterion):
        self.model.eval()
        xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y, dtype=torch.long).to(self.device)
        with torch.no_grad():
            loss = criterion(self.model(xt), yt)
        return loss.item()


def build_nn_classifiers() -> list[tuple[str, MLPClassifier]]:
    """Return a list of (name, MLPClassifier) for all configured sizes."""
    return [(name, MLPClassifier(name, layers)) for name, layers in MLP_CONFIGS.items()]
