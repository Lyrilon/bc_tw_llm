"""PyTorch neural network classifiers with streaming Parquet support.

Architecture catalogue (input_dim=4096, n_classes=5):
─────────────────────────────────────────────────────────────────────────────
 #  Name              Params (approx)  Type        Notes
─────────────────────────────────────────────────────────────────────────────
 1  MLP_Tiny          ~22 K            MLP         4096→64→5, sanity-check baseline
 2  MLP_Small         ~2.1 M           MLP         4096→512→5, single hidden layer
 3  MLP_Medium        ~4.3 M           MLP         4096→1024→5, wider single layer
 4  MLP_2L            ~2.2 M           MLP         4096→512→128→5, two hidden layers
 5  MLP_Deep          ~5.5 M           MLP         4096→1024→256→64→5, deep narrow
 6  MLP_Wide          ~17 M            MLP         4096→2048→1024→5, wide & shallow
 7  MLP_Bottleneck    ~6.4 M           MLP         4096→1024→128→1024→5, hourglass
 8  MLP_Residual      ~10 M            ResNet-MLP  4096→1024(×3 residual blocks)→5
 9  CNN_Tiny          ~135 K           1D-CNN      1-conv stage, 32 filters
10  CNN_Small         ~430 K           1D-CNN      2-conv stages, 64/128 filters
11  CNN_Medium        ~1.6 M           1D-CNN      3-conv stages, 64/128/256 filters
12  CNN_Deep          ~3.2 M           1D-CNN      4-conv stages, 64/128/256/512
13  CNN_Wide          ~5.6 M           1D-CNN      3-conv stages with wide filters (k=15)
14  Attention_Small   ~3.4 M           Self-Attn   4096→512, 4 heads, 2 transformer blocks
15  Attention_Medium  ~13 M            Self-Attn   4096→512, 8 heads, 4 transformer blocks
─────────────────────────────────────────────────────────────────────────────

All wrapped in NNClassifier: sklearn-compatible fit/predict + fit_streaming.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

log = logging.getLogger("classify")


# ---------------------------------------------------------------------------
# Streaming Parquet Datasets (unchanged)
# ---------------------------------------------------------------------------

class ParquetStreamDataset(Dataset):
    """Per-item Parquet reader — keeps only labels+offsets in RAM.

    Each __getitem__ reads one row from disk. Very low memory, slower I/O.
    """

    def __init__(self, parquet_dir: str, layer_index: int | None = None):
        import pyarrow.parquet as pq

        self.parquet_dir = Path(parquet_dir)
        files = sorted(self.parquet_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files in {parquet_dir}")

        self._files = files
        self._labels: list[int] = []
        self._file_idx: list[int] = []
        self._row_idx: list[int] = []

        for fi, f in enumerate(files):
            cols = ["label", "layer_index"] if layer_index is not None else ["label"]
            table = pq.read_table(f, columns=cols)
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

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        import pyarrow.parquet as pq
        fi, ri = self._file_idx[idx], self._row_idx[idx]
        table = pq.read_table(self._files[fi], columns=["hidden_state_vector"])
        vec = np.array(table.column("hidden_state_vector")[ri].as_py(), dtype=np.float32)
        return torch.from_numpy(vec), torch.tensor(self._labels[idx], dtype=torch.long)

    @property
    def labels(self) -> np.ndarray:
        return self._labels_np


class CachedParquetDataset(Dataset):
    """Loads all Parquet vectors into a contiguous numpy array on construction.

    Fast random access; uses ~6 GB RAM for 1.6 M × 4096 float32 vectors.
    """

    def __init__(self, parquet_dir: str, layer_index: int | None = None,
                 cache_dir: str | None = None):
        import pyarrow.parquet as pq

        self.parquet_dir = Path(parquet_dir)
        files = sorted(self.parquet_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files in {parquet_dir}")

        all_labels: list[np.ndarray] = []
        all_vectors: list[np.ndarray] = []

        for f in tqdm(files, desc="Loading parquet", leave=False):
            table = pq.read_table(f)
            df_chunk = table.to_pandas()

            if layer_index is not None:
                df_chunk = df_chunk[df_chunk["layer_index"] == layer_index]

            if len(df_chunk) == 0:
                continue

            all_labels.append(df_chunk["label"].values)
            all_vectors.append(np.vstack(df_chunk["hidden_state_vector"].values).astype(np.float32))
            del df_chunk, table

        self._labels = np.concatenate(all_labels)
        self._vectors = np.concatenate(all_vectors)
        log.info("CachedParquetDataset: %d samples, dim=%d", len(self), self._vectors.shape[1])

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self._vectors[idx]), torch.tensor(self._labels[idx], dtype=torch.long)

    @property
    def labels(self) -> np.ndarray:
        return self._labels


# ---------------------------------------------------------------------------
# Architecture 1 — MLP_Tiny
# ---------------------------------------------------------------------------

class MLP_Tiny(nn.Module):
    """Single tiny hidden layer MLP.

    Architecture : 4096 → 64 → 5
    Parameters   : ~22 K
    Purpose      : Absolute minimum-complexity baseline.  If even this can
                   achieve >20% (random chance), the signal exists.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Architecture 2 — MLP_Small
# ---------------------------------------------------------------------------

class MLP_Small(nn.Module):
    """Single hidden layer MLP, medium width.

    Architecture : 4096 → 512 → 5
    Parameters   : ~2.1 M
    Purpose      : Fast iteration baseline.  Good for verifying the training
                   pipeline and learning-rate range before committing to
                   heavier models.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Architecture 3 — MLP_Medium
# ---------------------------------------------------------------------------

class MLP_Medium(nn.Module):
    """Single wide hidden layer MLP.

    Architecture : 4096 → 1024 → 5
    Parameters   : ~4.3 M
    Purpose      : Tests whether width alone (without depth) is sufficient
                   to capture the signal.  Wide layers can memorise inter-
                   feature correlations in one shot.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Architecture 4 — MLP_2L
# ---------------------------------------------------------------------------

class MLP_2L(nn.Module):
    """Two hidden layer MLP, funnel shape.

    Architecture : 4096 → 512 → 128 → 5
    Parameters   : ~2.2 M
    Purpose      : Classic two-stage compression.  First layer captures broad
                   patterns, second layer refines into discriminative subspace.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Architecture 5 — MLP_Deep
# ---------------------------------------------------------------------------

class MLP_Deep(nn.Module):
    """Deep narrow MLP, four-stage funnel.

    Architecture : 4096 → 1024 → 256 → 64 → 5
    Parameters   : ~5.5 M
    Purpose      : Main classical workhorse.  Each stage halves the width
                   forcing progressive feature abstraction.  BatchNorm and
                   Dropout stabilise training on high-dimensional inputs.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Architecture 6 — MLP_Wide
# ---------------------------------------------------------------------------

class MLP_Wide(nn.Module):
    """Wide shallow MLP.

    Architecture : 4096 → 2048 → 1024 → 5
    Parameters   : ~17 M
    Purpose      : Tests width-over-depth hypothesis.  Very large first layer
                   can approximate many linear projections simultaneously.
                   Useful comparison against MLP_Deep of similar depth but
                   far more capacity.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Architecture 7 — MLP_Bottleneck
# ---------------------------------------------------------------------------

class MLP_Bottleneck(nn.Module):
    """Hourglass / autoencoder-style MLP with a 128-dim bottleneck.

    Architecture : 4096 → 1024 → 128 → 1024 → 5
    Parameters   : ~6.4 M
    Purpose      : Forces the network to compress the signal through a narrow
                   128-dim bottleneck then re-expand.  Acts as implicit
                   regularisation and tests whether a compact representation
                   is sufficient for classification.  Bottleneck activations
                   also serve as a learned feature extractor for inspection.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Architecture 8 — MLP_Residual
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Pre-activation residual block for 1-D feature vectors."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class MLP_Residual(nn.Module):
    """Residual MLP with three residual blocks in the 1024-dim latent space.

    Architecture : 4096 → 1024 → [ResBlock × 3] → 5
    Parameters   : ~10 M
    Purpose      : Residual connections allow gradients to flow cleanly
                   through depth, effectively training a deeper model without
                   vanishing-gradient issues.  LayerNorm (instead of BN) in
                   each block is more stable when the residual path carries
                   large activations.  Competitive with MLP_Wide at ~40% less
                   compute per step.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5,
                 hidden_dim: int = 1024, n_blocks: int = 3, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[_ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.head(self.blocks(self.proj(x)))


# ---------------------------------------------------------------------------
# Architecture 9 — CNN_Tiny
# ---------------------------------------------------------------------------

class CNN_Tiny(nn.Module):
    """Minimal 1D-CNN: single convolutional stage.

    Architecture : (B, 1, 4096) → Conv1d(1→32, k=7) → GAP → Linear(32→5)
    Parameters   : ~135 K
    Purpose      : Verify that local sliding-window patterns exist in the
                   hidden-state vector at all.  If this outperforms MLP_Tiny,
                   local structure is present.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


# ---------------------------------------------------------------------------
# Architecture 10 — CNN_Small
# ---------------------------------------------------------------------------

class CNN_Small(nn.Module):
    """Two-stage 1D-CNN.

    Architecture : (B,1,4096) → Conv(1→64,k=7) → Conv(64→128,k=5) → GAP → FC(128→5)
    Parameters   : ~430 K
    Purpose      : Two convolutional stages allow capturing features at two
                   different scales.  Stride-2 halves spatial resolution each
                   stage keeping computation manageable.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


# ---------------------------------------------------------------------------
# Architecture 11 — CNN_Medium
# ---------------------------------------------------------------------------

class CNN_Medium(nn.Module):
    """Three-stage 1D-CNN — the original default architecture.

    Architecture : Conv(1→64,k=7) → Conv(64→128,k=5) → Conv(128→256,k=3)
                   → GAP → FC(256→64) → FC(64→5)
    Parameters   : ~1.6 M
    Purpose      : Standard three-scale hierarchy.  The 256-channel bottleneck
                   after GAP + one FC refinement layer gives enough capacity
                   to learn non-linear class boundaries.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
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
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


# ---------------------------------------------------------------------------
# Architecture 12 — CNN_Deep
# ---------------------------------------------------------------------------

class CNN_Deep(nn.Module):
    """Four-stage 1D-CNN with max-pooling between stages.

    Architecture : Conv(1→64) → Conv(64→128) → Conv(128→256) → Conv(256→512)
                   → GAP → FC(512→128) → FC(128→5)
    Parameters   : ~3.2 M
    Purpose      : Extra depth extracts hierarchical features at four spatial
                   scales.  Max-pooling (instead of stride) preserves the most
                   activated positions which is useful when threat patterns are
                   sparse across the 4096 dimensions.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


# ---------------------------------------------------------------------------
# Architecture 13 — CNN_Wide
# ---------------------------------------------------------------------------

class CNN_Wide(nn.Module):
    """Three-stage 1D-CNN with large receptive fields (kernel=15).

    Architecture : Conv(1→128,k=15) → Conv(128→256,k=11) → Conv(256→512,k=7)
                   → GAP → FC(512→5)
    Parameters   : ~5.6 M
    Purpose      : Large kernels capture long-range dependencies within the
                   4096-dim vector (e.g. interactions between attention heads
                   spaced far apart in the vector).  Wide filters are
                   parameter-heavy but can learn patterns that small kernels
                   miss entirely.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


# ---------------------------------------------------------------------------
# Architecture 14 — Attention_Small
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    """Pre-LN Transformer block (Self-Attention + FFN)."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, seq_len, d_model)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class Attention_Small(nn.Module):
    """Small Transformer applied to 4096-dim vectors chunked into 8 tokens.

    Architecture : 4096 → project to (8 × 512) → 2 Transformer blocks
                   (4 heads, FFN=1024) → CLS-pool → FC(512→5)
    Parameters   : ~3.4 M
    Purpose      : Treats the 4096-dim hidden-state as a sequence of 8 tokens
                   of size 512 (mimicking 8 attention heads of Llama-3 8B).
                   Self-attention can model inter-head interactions that both
                   MLPs and CNNs cannot capture with a fixed receptive field.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5,
                 d_model: int = 512, n_heads: int = 4, n_blocks: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = input_dim // d_model  # e.g. 4096//512 = 8
        self.d_model = d_model
        # No explicit projection needed: reshape is the "projection"
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.blocks = nn.Sequential(*[
            _TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (B, input_dim) → (B, seq_len, d_model)
        B = x.size(0)
        x = x.view(B, self.seq_len, self.d_model) + self.pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)          # mean pooling over tokens
        return self.head(x)


# ---------------------------------------------------------------------------
# Architecture 15 — Attention_Medium
# ---------------------------------------------------------------------------

class Attention_Medium(nn.Module):
    """Medium Transformer: more heads, more depth, larger FFN.

    Architecture : 4096 → (8 × 512) → 4 Transformer blocks
                   (8 heads, FFN=2048) → mean-pool → FC(512→5)
    Parameters   : ~13 M
    Purpose      : Scales up Attention_Small with more expressive self-
                   attention (8 heads cover all 8 head-groups in Llama-3 8B)
                   and a deeper FFN per block.  The primary hypothesis is that
                   adversarial perturbations and identity forgery leave
                   structured cross-head signatures that stronger attention
                   can detect.
    """

    def __init__(self, input_dim: int = 4096, n_classes: int = 5,
                 d_model: int = 512, n_heads: int = 8, n_blocks: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = input_dim // d_model
        self.d_model = d_model
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.blocks = nn.Sequential(*[
            _TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.seq_len, self.d_model) + self.pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Model registry — ordered small → large
# ---------------------------------------------------------------------------

NN_CONFIGS: dict[str, dict] = {
    "MLP_Tiny":         {"cls": MLP_Tiny,         "dropout": 0.1},
    "MLP_Small":        {"cls": MLP_Small,         "dropout": 0.2},
    "MLP_Medium":       {"cls": MLP_Medium,        "dropout": 0.2},
    "MLP_2L":           {"cls": MLP_2L,            "dropout": 0.3},
    "MLP_Deep":         {"cls": MLP_Deep,          "dropout": 0.3},
    "MLP_Wide":         {"cls": MLP_Wide,          "dropout": 0.3},
    "MLP_Bottleneck":   {"cls": MLP_Bottleneck,    "dropout": 0.3},
    "MLP_Residual":     {"cls": MLP_Residual,      "dropout": 0.3},
    "CNN_Tiny":         {"cls": CNN_Tiny,           "dropout": 0.1},
    "CNN_Small":        {"cls": CNN_Small,          "dropout": 0.2},
    "CNN_Medium":       {"cls": CNN_Medium,         "dropout": 0.3},
    "CNN_Deep":         {"cls": CNN_Deep,           "dropout": 0.3},
    "CNN_Wide":         {"cls": CNN_Wide,           "dropout": 0.3},
    "Attention_Small":  {"cls": Attention_Small,    "dropout": 0.1},
    "Attention_Medium": {"cls": Attention_Medium,   "dropout": 0.1},
}


# ---------------------------------------------------------------------------
# Sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class NNClassifier:
    """Unified sklearn-compatible wrapper for all NN architectures.

    Usage::

        clf = NNClassifier("MLP_Deep", NN_CONFIGS["MLP_Deep"])
        clf.fit(X_train, y_train, X_val, y_val)
        y_pred = clf.predict(X_test)

        # Or streaming from disk:
        clf.fit_streaming(train_loader, val_loader)
        y_pred = clf.predict_loader(test_loader)
    """

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
        cls = self.config["cls"]
        dropout = self.config.get("dropout", 0.3)
        return cls(input_dim=input_dim, n_classes=n_classes, dropout=dropout)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        """Train from in-memory numpy arrays."""
        self.n_classes = int(y.max()) + 1
        self.model = self._build_model(X.shape[1], self.n_classes).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        log.info("%s: %d params, device=%s", self.name, total_params, self.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=3)
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=(self.device != "cpu"),
        )

        has_val = X_val is not None and y_val is not None
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        epoch_bar = tqdm(range(1, self.epochs + 1), desc=f"  {self.name}", unit="ep", leave=False)
        for epoch in epoch_bar:
            self.model.train()
            running_loss, n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimiser.step()
                running_loss += loss.item() * len(xb)
                n += len(xb)
            train_loss = running_loss / n

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
                        log.info("%s early-stopped at epoch %d (val=%.4f)",
                                 self.name, epoch, best_val_loss)
                        break
            else:
                epoch_bar.set_postfix(tr=f"{train_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def fit_streaming(self, train_loader: DataLoader,
                      val_loader: DataLoader | None = None):
        """Train from DataLoader (streaming from disk)."""
        sample_x, sample_y = next(iter(train_loader))
        input_dim = sample_x.shape[1]
        self.n_classes = int(sample_y.max()) + 1

        self.model = self._build_model(input_dim, self.n_classes).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        n_train = len(train_loader.dataset) if hasattr(train_loader.dataset, "__len__") else -1
        log.info("%s: %d params, %d train samples, device=%s",
                 self.name, total_params, n_train, self.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=3)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss, n = 0.0, 0
            batch_bar = tqdm(train_loader, desc=f"  {self.name} ep{epoch}", leave=False)
            for xb, yb in batch_bar:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimiser.step()
                running_loss += loss.item() * len(xb)
                n += len(xb)
                batch_bar.set_postfix(loss=f"{running_loss/n:.4f}")
            train_loss = running_loss / n

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
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False,
        )
        with torch.no_grad():
            for (xb,) in loader:
                preds.append(self.model(xb.to(self.device)).argmax(1).cpu())
        return torch.cat(preds).numpy()

    def predict_loader(self, loader: DataLoader) -> np.ndarray:
        """Predict from DataLoader (streaming)."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(self.model(xb.to(self.device)).argmax(1).cpu())
        return torch.cat(preds).numpy()

    def _compute_loss(self, X, y, criterion):
        self.model.eval()
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.long)),
            batch_size=self.batch_size, shuffle=False,
        )
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                total += criterion(self.model(xb), yb).item() * len(xb)
                n += len(xb)
        return total / n

    def _compute_loss_loader(self, loader, criterion):
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                total += criterion(self.model(xb), yb).item() * len(xb)
                n += len(xb)
        return total / n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_nn_classifiers(
    names: list[str] | None = None,
    **kwargs,
) -> list[tuple[str, NNClassifier]]:
    """Return (name, NNClassifier) pairs.

    Args:
        names: Subset of model names to build.  None = all models.
        **kwargs: Passed to NNClassifier (lr, epochs, batch_size, patience, device).
    """
    registry = NN_CONFIGS if names is None else {n: NN_CONFIGS[n] for n in names}
    return [(name, NNClassifier(name, config, **kwargs))
            for name, config in registry.items()]
