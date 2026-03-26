"""Quick experiment script for testing individual neural network architectures.

Edit the Config class below to choose a model and tune hyperparameters,
then run:

    python scripts/test_nn.py

Available model names (ordered small → large):
    CNN_Tiny, CNN_Small, CNN_Medium, CNN_Deep, CNN_Wide,
    MLP_Tiny, MLP_Small, MLP_2L, MLP_Medium, MLP_Bottleneck,
    MLP_Deep, Attention_Small, MLP_Wide, MLP_Residual, Attention_Medium
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make sure the project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from classify.nn_models import NN_CONFIGS, NNClassifier, CachedParquetDataset
from classify.logging_setup import setup_logging, LogContext


# ===========================================================================
# Config — edit this to run your experiment
# ===========================================================================

@dataclass
class Config:
    # ── Data ────────────────────────────────────────────────────────────────
    data_dir: str = "/root/autodl-tmp/data/"

    # Layer to train on. None = all layers mixed (cross-layer).
    layer_index: int | None = None

    # Stratified sample size loaded into memory. 0 = full dataset.
    # Recommended: 100_000 for quick iteration, 0 for full training.
    sample_size: int = 100_000

    # ── Model ───────────────────────────────────────────────────────────────
    # Choose one name from the list at the top of this file.
    model_name: str = "MLP_Deep"

    # ── Training ────────────────────────────────────────────────────────────
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 512
    patience: int = 8          # early-stopping patience (epochs without val improvement)
    dropout: float | None = None  # None = use model default from NN_CONFIGS

    # ── Split ───────────────────────────────────────────────────────────────
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42

    # ── Output ──────────────────────────────────────────────────────────────
    # Relative to project root
    log_dir: str = "logs"
    results_dir: str = "scripts/results"


# ===========================================================================
# Helpers
# ===========================================================================

LABEL_MAP = {
    0: "honest",
    1: "silent_precision_downgrade",
    2: "identity_forgery",
    3: "random_noise",
    4: "adversarial_perturbation",
}


def _load_data(cfg: Config, ctx: LogContext):
    """Load (and optionally subsample) the dataset into memory."""
    import pandas as pd

    p = Path(cfg.data_dir)
    files = sorted(p.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {cfg.data_dir}")

    with ctx.group("Loading data"):
        ctx.step(f"Found {len(files)} parquet files in {cfg.data_dir}")
        df = pd.read_parquet(files)
        ctx.step(f"Total records: {len(df):,}")

        if cfg.layer_index is not None:
            df = df[df["layer_index"] == cfg.layer_index].reset_index(drop=True)
            ctx.step(f"Filtered to layer {cfg.layer_index}: {len(df):,} records")

        if cfg.sample_size > 0 and len(df) > cfg.sample_size:
            frac = cfg.sample_size / len(df)
            parts = [g.sample(frac=frac, random_state=cfg.seed)
                     for _, g in df.groupby("label")]
            df = pd.concat(parts).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
            ctx.step(f"Sampled {len(df):,} records (stratified)", last=True)
        else:
            ctx.step(f"Using full dataset ({len(df):,} records)", last=True)

    return df


def _extract_vectors(df, ctx: LogContext):
    """Stack hidden_state_vector column into a numpy array."""
    from tqdm import tqdm

    n = len(df)
    dim = len(df["hidden_state_vector"].iloc[0])
    X = np.zeros((n, dim), dtype=np.float32)
    batch = 50_000
    for i in tqdm(range(0, n, batch), desc="  Stacking vectors", leave=False):
        end = min(i + batch, n)
        X[i:end] = np.vstack(df["hidden_state_vector"].values[i:end])
    y = df["label"].values.astype(np.int64)
    return X, y


# ===========================================================================
# Main
# ===========================================================================

def main():
    cfg = Config()

    # ── Validate model name ─────────────────────────────────────────────────
    if cfg.model_name not in NN_CONFIGS:
        available = ", ".join(NN_CONFIGS.keys())
        raise ValueError(
            f"Unknown model '{cfg.model_name}'.\nAvailable: {available}"
        )

    # ── Setup logging ───────────────────────────────────────────────────────
    project_root = Path(__file__).parent.parent
    setup_logging(str(project_root / cfg.log_dir))
    ctx = LogContext()

    results_dir = project_root / cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Header ──────────────────────────────────────────────────────────────
    layer_str = str(cfg.layer_index) if cfg.layer_index is not None else "all (cross-layer)"
    ctx.section(
        f"NN Experiment: {cfg.model_name}",
        f"layer={layer_str} | sample={cfg.sample_size or 'full'} | "
        f"lr={cfg.lr} | epochs={cfg.epochs} | bs={cfg.batch_size} | patience={cfg.patience}"
    )

    # ── Model info ──────────────────────────────────────────────────────────
    model_cfg = dict(NN_CONFIGS[cfg.model_name])
    if cfg.dropout is not None:
        model_cfg["dropout"] = cfg.dropout

    import torch
    tmp = NNClassifier(cfg.model_name, model_cfg)
    tmp.model = tmp._build_model(4096, 5)
    n_params = sum(p.numel() for p in tmp.model.parameters())
    del tmp

    ctx.step(f"Model: {cfg.model_name}  ({n_params:,} parameters)")
    ctx.step(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # ── Load data ───────────────────────────────────────────────────────────
    df = _load_data(cfg, ctx)

    with ctx.group("Feature extraction"):
        t0 = time.time()
        X, y = _extract_vectors(df, ctx)
        del df
        ctx.step(f"Shape: {X.shape[0]:,} × {X.shape[1]}  ({time.time()-t0:.1f}s)", last=True)

    # ── Split ───────────────────────────────────────────────────────────────
    idx_tv, idx_test = train_test_split(
        np.arange(len(y)), test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )
    relative_val = cfg.val_size / (1 - cfg.test_size)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=relative_val, random_state=cfg.seed, stratify=y[idx_tv]
    )
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    ctx.step(f"Split: train={len(y_train):,} / val={len(y_val):,} / test={len(y_test):,}")

    # ── Train ───────────────────────────────────────────────────────────────
    clf = NNClassifier(
        cfg.model_name, model_cfg,
        lr=cfg.lr,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        patience=cfg.patience,
    )

    with ctx.group(f"Training {cfg.model_name}"):
        t0 = time.time()
        clf.fit(X_train, y_train, X_val, y_val)
        elapsed = time.time() - t0
        ctx.step(f"Done in {elapsed:.1f}s", last=True)

    # ── Evaluate ────────────────────────────────────────────────────────────
    with ctx.group("Evaluation", last=True):
        y_val_pred = clf.predict(X_val)
        y_test_pred = clf.predict(X_test)

        f1_val  = f1_score(y_val,  y_val_pred,  average="weighted", zero_division=0)
        f1_test = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
        acc_val  = (y_val  == y_val_pred).mean()
        acc_test = (y_test == y_test_pred).mean()

        ctx.step(f"Val:  acc={acc_val:.4f}  F1w={f1_val:.4f}")
        ctx.step(f"Test: acc={acc_test:.4f}  F1w={f1_test:.4f}", last=True)

    # ── Classification report ───────────────────────────────────────────────
    class_names = [LABEL_MAP[i] for i in sorted(set(y_test.tolist()))]
    report = classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0)

    ctx.blank()
    ctx.text("Test Classification Report")
    ctx.text("─" * 60)
    ctx.text(report)

    # ── Save report ─────────────────────────────────────────────────────────
    layer_tag = f"layer{cfg.layer_index}" if cfg.layer_index is not None else "cross"
    report_path = results_dir / f"{cfg.model_name}_{layer_tag}_report.txt"
    report_path.write_text(
        f"Model: {cfg.model_name}\n"
        f"Params: {n_params:,}\n"
        f"Layer: {layer_str}\n"
        f"Sample: {cfg.sample_size or 'full'}\n"
        f"lr={cfg.lr}  epochs={cfg.epochs}  bs={cfg.batch_size}  patience={cfg.patience}\n"
        f"Val  acc={acc_val:.4f}  F1w={f1_val:.4f}\n"
        f"Test acc={acc_test:.4f}  F1w={f1_test:.4f}\n\n"
        + report
    )
    ctx.text(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
