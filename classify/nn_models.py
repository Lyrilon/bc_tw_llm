"""PyTorch neural network classifiers with streaming Parquet support.

Three architectures:
  1. MLP_Light   – 4096 → 512 → 5  (pipeline validator)
  2. MLP_Deep    – 4096 → 1024 → 256 → 64 → 5  (main workhorse)
  3. CNN_1D      – Conv1d on 4096-dim vectors (local pattern extractor)

All wrapped in an sklearn-compatible interface (fit/predict).
Supports both in-memory numpy arrays and streaming from Parquet files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

log = logging.getLogger("classify")


# ---------------------------------------------------------------------------
# Streaming Parquet Dataset
# ---------------------------------------------------------------------------

class ParquetStreamDataset(Dataset):
    """PyTorch Dataset that reads hidden-state vectors from Parquet files on demand.

    Keeps only metadata (labels, file offsets) in memory.  Vectors are read
    from disk per-batch, keeping RAM usage constant regardless of dataset size.
    """

    def __init__(self, parquet_dir: str, layer_index: int | None = None):
        import pyarrow.parquet as pq

        self.parquet_dir = Path(parquet_dir)
        files = sorted(self.parquet_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files in {parquet_dir}")

        # Build index: (file_idx, row_within_file) for each sample
        self._files = files
        self._file_tables = []  # lazy-loaded
        self._labels: list[int] = []
        self._file_idx: list[int] = []
        self._row_idx: list[int] = []

        for fi, f in enumerate(files):
            table = pq.read_table(f, columns=["label", "layer_index"] if layer_index is not None else ["label"])
            labels = table.column("label").to_pylist()

            if layer_index is not None:
                layers = table.column("layer_index").to_pylist()
                for ri, (lbl, li) in enumerate(zip(labels, layers)):
                    if li == layer_index:
                        self._labels.append(lbl)
                        self._file_idx.append(fi)
                        self._row_idx.append(ri)
            else:
                for ri, lbl in enumerate(labels):
                    self._labels.append(lbl)
                    self._file_idx.append(fi)
                    self._row_idx.append(ri)
            del table

        self._labels_np = np.array(self._labels, dtype=np.int64)
        log.info("ParquetStreamDataset: %d samples from %d files", len(self), len(files))

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        import pyarrow.parquet as pq

        fi = self._file_idx[idx]
        ri = self._row_idx[idx]

        # Read single row's hidden_state_vector
        table = pq.read_table(self._files[fi], columns=["hidden_state_vector"])
        vec = np.array(table.column("hidden_state_vector")[ri].as_py(), dtype=np.float32)
        label = self._labels[idx]

        return torch.from_numpy(vec), torch.tensor(label, dtype=torch.long)

    @property
    def labels(self) -> np.ndarray:
        return self._labels_np


class CachedParquetDataset(Dataset):
    """Memory-mapped variant: reads all vectors once into a numpy memmap for fast access.

    Use this when you can afford ~6GB disk for the memmap cache but want
    to avoid re-reading Parquet per item.
    """

    def __init__(self, parquet_dir: str, layer_index: int | None = None,
                 cache_dir: str | None = None):
        import pyarrow.parquet as pq

        self.parquet_dir = Path(parquet_dir)
        files = sorted(self.parquet_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files in {parquet_dir}")

        # First pass: collect labels and count samples
        all_labels: list[int] = []
        all_vectors: list[np.ndarray] = []

        for f in tqdm(files, desc="Loading parquet"):
            table = pq.read_table(f)
            df_chunk = table.to_pandas()

            if layer_index is not None:
                mask = df_chunk["layer_index"] == layer_index
                df_chunk = df_chunk[mask]

            if len(df_chunk) == 0:
                continue

            labels = df_chunk["label"].values
            vectors = np.vstack(df_chunk["hidden_state_vector"].values).astype(np.float32)
            all_labels.append(labels)
            all_vectors.append(vectors)
            del df_chunk, table

        self._labels = np.concatenate(all_labels)
        self._vectors = np.concatenate(all_vectors)
        log.info("CachedParquetDataset: %d samples, dim=%d", len(self), self._vectors.shape[1])

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        vec = torch.from_numpy(self._vectors[idx])
        label = torch.tensor(self._labels[idx], dtype=torch.long)
        return vec, label

    @property
    def labels(self) -> np.ndarray:
        return self._labels


# ---------------------------------------------------------------------------
# Neural network architectures
# ---------------------------------------------------------------------------

def _build_mlp(input_dim: int, n_classes: int, hidden_layers: list[int],
               dropout: float = 0.3) -> nn.Sequential:
    """Build MLP: [Linear → BN → ReLU → Dropout] × N → Linear."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_layers:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


class CNN1DClassifier(nn.Module):
    """1D-CNN that treats the 4096-dim vector as a 1D signal.

    Slides conv filters across the hidden-state dimensions to capture
    local activation patterns from attention heads / neuron groups.
    """

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        # Treat input as (batch, 1, input_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (batch, 256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, input_dim) → (batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

NN_CONFIGS: dict[str, dict] = {
    "MLP_Light": {
        "type": "mlp",
        "hidden_layers": [512],
        "dropout": 0.2,
        "description": "4096→512→5 (pipeline validator)",
    },
    "MLP_Deep": {
        "type": "mlp",
        "hidden_layers": [1024, 256, 64],
        "dropout": 0.3,
        "description": "4096→1024→256→64→5 (main workhorse)",
    },
    "CNN_1D": {
        "type": "cnn1d",
        "dropout": 0.3,
        "description": "Conv1d local pattern extractor",
    },
}


# ---------------------------------------------------------------------------
# Sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class NNClassifier:
    """Unified wrapper for all NN architectures with sklearn-like fit/predict."""

    def __init__(
        self,
        name: str,
        config: dict,
        *,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 512,
        patience: int = 8,
        device: str | None = None,
    ):
        self.name = name
        self.config = config
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.n_classes: int = 0

    def _build_model(self, input_dim: int, n_classes: int) -> nn.Module:
        cfg = self.config
        if cfg["type"] == "mlp":
            return _build_mlp(input_dim, n_classes, cfg["hidden_layers"], cfg.get("dropout", 0.3))
        elif cfg["type"] == "cnn1d":
            return CNN1DClassifier(input_dim, n_classes, cfg.get("dropout", 0.3))
        else:
            raise ValueError(f"Unknown model type: {cfg['type']}")

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        """Train from in-memory numpy arrays."""
        self.n_classes = int(y.max()) + 1
        self.model = self._build_model(X.shape[1], self.n_classes).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        log.info("%s: %d params, device=%s", self.name, total_params, self.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=3, verbose=False
        )
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=(self.device != "cpu"))

        has_val = X_val is not None and y_val is not None
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        epoch_bar = tqdm(range(1, self.epochs + 1), desc=f"  {self.name}", unit="ep", leave=False)
        for epoch in epoch_bar:
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

            if has_val:
                val_loss = self._compute_loss(X_val, y_val, criterion)
                scheduler.step(val_loss)
                epoch_bar.set_postfix(tr=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        log.info("%s early-stopped at epoch %d (val=%.4f)", self.name, epoch, best_val_loss)
                        break
            else:
                epoch_bar.set_postfix(tr=f"{train_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def fit_streaming(self, train_loader: DataLoader, val_loader: DataLoader | None = None):
        """Train from DataLoader (streaming from disk)."""
        # Peek at first batch to get dimensions
        sample_x, sample_y = next(iter(train_loader))
        input_dim = sample_x.shape[1]
        self.n_classes = int(sample_y.max()) + 1

        # Count total samples for logging
        if hasattr(train_loader.dataset, '__len__'):
            n_train = len(train_loader.dataset)
        else:
            n_train = -1

        self.model = self._build_model(input_dim, self.n_classes).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        log.info("%s: %d params, %d train samples, device=%s",
                 self.name, total_params, n_train, self.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=3, verbose=False
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            n_samples = 0
            batch_bar = tqdm(train_loader, desc=f"  {self.name} ep{epoch}", leave=False)
            for xb, yb in batch_bar:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimiser.step()
                running_loss += loss.item() * len(xb)
                n_samples += len(xb)
                batch_bar.set_postfix(loss=f"{running_loss/n_samples:.4f}")
            train_loss = running_loss / n_samples

            # Validation
            if val_loader is not None:
                val_loss = self._compute_loss_loader(val_loader, criterion)
                scheduler.step(val_loss)
                log.info("  epoch %d/%d — train=%.4f val=%.4f",
                         epoch, self.epochs, train_loss, val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        log.info("%s early-stopped at epoch %d (val=%.4f)",
                                 self.name, epoch, best_val_loss)
                        break
            else:
                log.info("  epoch %d/%d — train=%.4f", epoch, self.epochs, train_loss)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from in-memory numpy array."""
        self.model.eval()
        preds = []
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                preds.append(logits.argmax(dim=1).cpu())
        return torch.cat(preds).numpy()

    def predict_loader(self, loader: DataLoader) -> np.ndarray:
        """Predict from DataLoader (streaming)."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                preds.append(logits.argmax(dim=1).cpu())
        return torch.cat(preds).numpy()

    def _compute_loss(self, X, y, criterion):
        self.model.eval()
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = criterion(self.model(xb), yb)
                total_loss += loss.item() * len(xb)
                n += len(xb)
        return total_loss / n

    def _compute_loss_loader(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = criterion(self.model(xb), yb)
                total_loss += loss.item() * len(xb)
                n += len(xb)
        return total_loss / n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_nn_classifiers(**kwargs) -> list[tuple[str, NNClassifier]]:
    """Return (name, NNClassifier) for all configured architectures."""
    return [
        (name, NNClassifier(name, config, **kwargs))
        for name, config in NN_CONFIGS.items()
    ]
