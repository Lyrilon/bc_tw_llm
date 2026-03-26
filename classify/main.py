"""
ML classification of LLM hidden-state vectors.

Two task modes (both predict threat label 0-4):

  - ``per-layer``   – train one classifier **per decoder layer**; each
                      classifier only sees hidden states from its own layer.
  - ``cross-layer`` – train a **single** classifier on hidden states from
                      all layers mixed together (layer origin is unknown).

Usage::

    python -m classify --data-dir output --task per-layer
    python -m classify --data-dir output --task cross-layer
"""

from __future__ import annotations

import argparse
import logging
import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .nn_models import build_nn_classifiers, NNClassifier, NN_CONFIGS, CachedParquetDataset
from .sampling import stratified_sample
from .logging_setup import setup_logging, LogContext

log = logging.getLogger("classify")

# Threat label definitions (mirrors threat_dataset.config.LABEL_MAP)
LABEL_MAP: dict[int, str] = {
    0: "honest",
    1: "silent_precision_downgrade",
    2: "identity_forgery",
    3: "random_noise",
    4: "adversarial_perturbation",
}

# Module-level LogContext, initialised in main()
ctx: LogContext = LogContext()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_dir: str) -> pd.DataFrame:
    """Load all Parquet part files from *data_dir*."""
    p = Path(data_dir)
    files = sorted(p.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")
    df = pd.read_parquet(files)
    ctx.step(f"Loaded {len(df):,} records from {len(files)} files")
    return df


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def _extract_X_y(df: pd.DataFrame, pca_dim: int | None, batch_size: int = 50000,
                 label: str = ""):
    """Extract feature matrix and threat labels. Returns (X, y, class_names, pca)."""
    y = df["label"].values
    class_names = [LABEL_MAP[i] for i in sorted(df["label"].unique())]
    n_samples = len(df)

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    pca = None
    if pca_dim is not None:
        pca = IncrementalPCA(n_components=pca_dim)

        iterator = tqdm(range(0, n_samples, batch_size), desc=f"    PCA fitting ({label})" if label else "    PCA fitting", leave=False) if use_tqdm else range(0, n_samples, batch_size)
        for i in iterator:
            end = min(i + batch_size, n_samples)
            X_batch = np.vstack(df["hidden_state_vector"].values[i:end]).astype(np.float32)
            pca.partial_fit(X_batch)
            del X_batch

        X = np.zeros((n_samples, pca_dim), dtype=np.float32)
        iterator = tqdm(range(0, n_samples, batch_size), desc=f"    PCA transform ({label})" if label else "    PCA transform", leave=False) if use_tqdm else range(0, n_samples, batch_size)
        for i in iterator:
            end = min(i + batch_size, n_samples)
            X_batch = np.vstack(df["hidden_state_vector"].values[i:end]).astype(np.float32)
            X[i:end] = pca.transform(X_batch)
            del X_batch
    else:
        vec_dim = len(df["hidden_state_vector"].iloc[0])
        X = np.zeros((n_samples, vec_dim), dtype=np.float32)
        iterator = tqdm(range(0, n_samples, batch_size), desc=f"    Stacking ({label})" if label else "    Stacking", leave=False) if use_tqdm else range(0, n_samples, batch_size)
        for i in iterator:
            end = min(i + batch_size, n_samples)
            X[i:end] = np.vstack(df["hidden_state_vector"].values[i:end]).astype(np.float32)

    return X, y, class_names, pca


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def _split(X, y, val_size: float, test_size: float, seed: int):
    """Stratified train / val / test split."""
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=relative_val, random_state=seed, stratify=y_tv,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def _build_classifiers(n_classes: int) -> list[tuple[str, object]]:
    """Return a list of (name, estimator) pairs."""
    clfs: list[tuple[str, object]] = [
        ("LogisticRegression", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, n_jobs=-1),
        )),
        ("RandomForest", RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42,
        )),
        ("KNN", make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        )),
    ]

    try:
        import lightgbm as lgb
        clfs.append(("LightGBM", lgb.LGBMClassifier(
            n_estimators=200, num_leaves=63, learning_rate=0.1,
            objective="multiclass", num_class=n_classes,
            n_jobs=-1, random_state=42, verbose=-1,
        )))
    except ImportError:
        pass  # silently skip, will note in classifier count

    return clfs


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate(name, clf, X, y, class_names, split_label):
    """Predict and return a metrics dict + predictions."""
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    f1_w = f1_score(y, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y, y_pred, average="macro", zero_division=0)
    return {
        "classifier": name,
        "split": split_label,
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
    }, y_pred


def _save_confusion_matrix(y_true, y_pred, class_names, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.7), max(5, len(class_names) * 0.6)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Models directory (project root / models)
MODELS_DIR = Path(__file__).parent.parent / "models"


def _save_best_model(clf, name: str, tag: str, f1: float, is_nn: bool):
    """Save model weights if it's the best test F1 for this tag.

    Saves to models/{tag}/best_model.* with a metadata JSON.
    Only overwrites if f1 > previous best.
    """
    import json

    tag_dir = MODELS_DIR / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    meta_path = tag_dir / "best_meta.json"

    # Check if current best exists
    if meta_path.exists():
        prev = json.loads(meta_path.read_text())
        if prev.get("f1_weighted", 0) >= f1:
            return False  # not better

    # Save model
    if is_nn:
        model_path = tag_dir / "best_model.pt"
        torch.save(clf.model.state_dict(), model_path)
    else:
        model_path = tag_dir / "best_model.joblib"
        joblib.dump(clf, model_path)

    # Save metadata
    meta = {
        "classifier": name,
        "tag": tag,
        "f1_weighted": f1,
        "model_file": model_path.name,
        "is_nn": is_nn,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    ctx.step(f"★ Best model saved → {model_path.relative_to(MODELS_DIR.parent)}")
    return True


def _run_classifiers(
    X_train_pca, X_val_pca, X_test_pca,
    X_train_raw, X_val_raw, X_test_raw,
    y_train, y_val, y_test,
    class_names, results_dir: Path, tag: str,
):
    """Train all classifiers, evaluate, save artifacts. Return metrics list."""
    n_classes = len(class_names)
    all_metrics: list[dict] = []

    # --- Classical ML classifiers (use PCA data) ---
    classical_clfs = _build_classifiers(n_classes)
    n_classical = len(classical_clfs)

    # --- Neural Network classifiers (use RAW data, NO PCA) ---
    nn_clfs = build_nn_classifiers()
    n_nn = len(nn_clfs)

    # -- Classical ML --
    is_last_classical_group = (n_nn == 0)
    with ctx.group(f"Classical ML ({n_classical} models, PCA {X_train_pca.shape[1]}-dim)",
                   last=is_last_classical_group):
        for i, (name, clf) in enumerate(classical_clfs, 1):
            is_last_clf = (i == n_classical)
            with ctx.group(f"[{i}/{n_classical}] {name}", last=is_last_clf):
                t0 = time.time()
                clf.fit(X_train_pca, y_train)
                elapsed = time.time() - t0
                ctx.step(f"Training...done ({elapsed:.1f}s)")

                m_val, y_pred_val = _evaluate(name, clf, X_val_pca, y_val, class_names, "val")
                m_val["train_time_s"] = elapsed
                m_val["tag"] = tag
                all_metrics.append(m_val)
                ctx.step(f"Val:  acc={m_val['accuracy']:.4f}  F1w={m_val['f1_weighted']:.4f}  F1m={m_val['f1_macro']:.4f}")

                _save_confusion_matrix(
                    y_val, y_pred_val, class_names,
                    f"{name} — val ({tag})",
                    results_dir / f"cm_{tag}_{name}_val.png",
                )

                m_test, y_pred_test = _evaluate(name, clf, X_test_pca, y_test, class_names, "test")
                m_test["train_time_s"] = elapsed
                m_test["tag"] = tag
                all_metrics.append(m_test)
                ctx.step(f"Test: acc={m_test['accuracy']:.4f}  F1w={m_test['f1_weighted']:.4f}  F1m={m_test['f1_macro']:.4f}")

                _save_best_model(clf, name, tag, m_test["f1_weighted"], is_nn=False)

                _save_confusion_matrix(
                    y_test, y_pred_test, class_names,
                    f"{name} — test ({tag})",
                    results_dir / f"cm_{tag}_{name}_test.png",
                )

                report = classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0)
                (results_dir / f"report_{tag}_{name}_test.txt").write_text(report)

    # -- Neural Networks --
    if n_nn > 0:
        with ctx.group(f"Neural Networks ({n_nn} models, RAW {X_train_raw.shape[1]}-dim)", last=True):
            for i, (name, clf) in enumerate(nn_clfs, 1):
                is_last_nn = (i == n_nn)
                with ctx.group(f"[{i}/{n_nn}] {name}", last=is_last_nn):
                    t0 = time.time()
                    clf.fit(X_train_raw, y_train, X_val_raw, y_val)
                    elapsed = time.time() - t0
                    ctx.step(f"Training...done ({elapsed:.1f}s)")

                    m_val, y_pred_val = _evaluate(name, clf, X_val_raw, y_val, class_names, "val")
                    m_val["train_time_s"] = elapsed
                    m_val["tag"] = tag
                    all_metrics.append(m_val)
                    ctx.step(f"Val:  acc={m_val['accuracy']:.4f}  F1w={m_val['f1_weighted']:.4f}  F1m={m_val['f1_macro']:.4f}")

                    _save_confusion_matrix(
                        y_val, y_pred_val, class_names,
                        f"{name} — val ({tag})",
                        results_dir / f"cm_{tag}_{name}_val.png",
                    )

                    m_test, y_pred_test = _evaluate(name, clf, X_test_raw, y_test, class_names, "test")
                    m_test["train_time_s"] = elapsed
                    m_test["tag"] = tag
                    all_metrics.append(m_test)
                    ctx.step(f"Test: acc={m_test['accuracy']:.4f}  F1w={m_test['f1_weighted']:.4f}  F1m={m_test['f1_macro']:.4f}")

                    _save_best_model(clf, name, tag, m_test["f1_weighted"], is_nn=True)

                    _save_confusion_matrix(
                        y_test, y_pred_test, class_names,
                        f"{name} — test ({tag})",
                        results_dir / f"cm_{tag}_{name}_test.png",
                    )

                    report = classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0)
                    (results_dir / f"report_{tag}_{name}_test.txt").write_text(report)

    return all_metrics


# ---------------------------------------------------------------------------
# Task: cross-layer  (one model for all layers)
# ---------------------------------------------------------------------------


def run_cross_layer(df: pd.DataFrame, pca_dim, val_size, test_size, seed, results_dir):
    """Single classifier trained on all layers mixed together."""

    # Feature extraction
    with ctx.group("Feature extraction"):
        t0 = time.time()
        X_raw, y, class_names, _ = _extract_X_y(df, pca_dim=None, label="raw")
        elapsed_raw = time.time() - t0
        ctx.step(f"Raw features: {X_raw.shape[0]:,} x {X_raw.shape[1]:,} ({elapsed_raw:.1f}s)")

        t0 = time.time()
        X_pca, _, _, pca_obj = _extract_X_y(df, pca_dim=pca_dim, label="pca")
        elapsed_pca = time.time() - t0
        variance = pca_obj.explained_variance_ratio_.sum() * 100 if pca_obj else 0
        ctx.step(f"PCA features: {X_pca.shape[0]:,} x {X_pca.shape[1]:,} (variance: {variance:.1f}%, {elapsed_pca:.1f}s)", last=True)

    # Split
    idx_tv, idx_test = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
    )
    relative_val = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=relative_val, random_state=seed, stratify=y[idx_tv]
    )

    X_train_raw, X_val_raw, X_test_raw = X_raw[idx_train], X_raw[idx_val], X_raw[idx_test]
    X_train_pca, X_val_pca, X_test_pca = X_pca[idx_train], X_pca[idx_val], X_pca[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    ctx.step(f"Data split: train={len(y_train):,} / val={len(y_val):,} / test={len(y_test):,}")

    return _run_classifiers(
        X_train_pca, X_val_pca, X_test_pca,
        X_train_raw, X_val_raw, X_test_raw,
        y_train, y_val, y_test,
        class_names, results_dir, tag="cross_layer"
    )


# ---------------------------------------------------------------------------
# Task: per-layer  (one model per decoder layer)
# ---------------------------------------------------------------------------


def run_per_layer(df: pd.DataFrame, pca_dim, val_size, test_size, seed, results_dir):
    """One classifier per decoder layer, each only sees its own layer's data."""
    layers = sorted(df["layer_index"].unique())
    n_layers = len(layers)
    ctx.step(f"Found {n_layers} decoder layers")

    all_metrics: list[dict] = []

    for li, layer_idx in enumerate(layers, 1):
        layer_df = df[df["layer_index"] == layer_idx]
        tag = f"layer_{layer_idx}"
        is_last_layer = (li == n_layers)

        with ctx.group(f"[{li}/{n_layers}] Layer {layer_idx} ({len(layer_df):,} samples)",
                       last=is_last_layer):
            # Extract features
            with ctx.group("Feature extraction"):
                t0 = time.time()
                X_raw, y, class_names, _ = _extract_X_y(layer_df, pca_dim=None, label=f"L{layer_idx} raw")
                elapsed_raw = time.time() - t0
                ctx.step(f"Raw: {X_raw.shape[0]:,} x {X_raw.shape[1]:,} ({elapsed_raw:.1f}s)")

                t0 = time.time()
                X_pca, _, _, pca_obj = _extract_X_y(layer_df, pca_dim=pca_dim, label=f"L{layer_idx} pca")
                elapsed_pca = time.time() - t0
                variance = pca_obj.explained_variance_ratio_.sum() * 100 if pca_obj else 0
                ctx.step(f"PCA: {X_pca.shape[0]:,} x {X_pca.shape[1]:,} (variance: {variance:.1f}%, {elapsed_pca:.1f}s)", last=True)

            # Skip if too few
            unique, counts = np.unique(y, return_counts=True)
            if counts.min() < 3:
                ctx.step(f"Skipped: too few samples (min class={counts.min()})", last=True)
                continue

            # Split
            splits_raw = _split(X_raw, y, val_size, test_size, seed)
            splits_pca = _split(X_pca, y, val_size, test_size, seed)
            X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = splits_raw
            X_train_pca, X_val_pca, X_test_pca, _, _, _ = splits_pca

            ctx.step(f"Data split: train={len(y_train):,} / val={len(y_val):,} / test={len(y_test):,}")

            layer_dir = results_dir / tag
            layer_dir.mkdir(parents=True, exist_ok=True)

            metrics = _run_classifiers(
                X_train_pca, X_val_pca, X_test_pca,
                X_train_raw, X_val_raw, X_test_raw,
                y_train, y_val, y_test,
                class_names, layer_dir, tag=tag
            )
            all_metrics.extend(metrics)

    return all_metrics


# ---------------------------------------------------------------------------
# Track 2: Streaming Neural Network pipeline (full data, no PCA)
# ---------------------------------------------------------------------------


def run_nn_streaming(data_dir: str, task: str, val_size: float, test_size: float,
                     seed: int, results_dir: Path, nn_kwargs: dict):
    """Train neural networks on full-resolution data via streaming DataLoaders."""
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import train_test_split

    layer_indices = [None] if task == "cross-layer" else None

    if task == "per-layer":
        import pyarrow.parquet as pq
        p = Path(data_dir)
        first_file = sorted(p.glob("*.parquet"))[0]
        table = pq.read_table(first_file, columns=["layer_index"])
        layer_indices = sorted(set(table.column("layer_index").to_pylist()))
        del table
        ctx.step(f"Found {len(layer_indices)} layers for streaming")

    all_metrics: list[dict] = []
    n_groups = len(layer_indices)

    for gi, layer_idx in enumerate(layer_indices, 1):
        tag = "cross_layer" if layer_idx is None else f"layer_{layer_idx}"
        is_last_group = (gi == n_groups)
        group_title = "All layers (cross-layer)" if layer_idx is None else f"[{gi}/{n_groups}] Layer {layer_idx}"

        with ctx.group(group_title, last=is_last_group):
            t0 = time.time()
            dataset = CachedParquetDataset(data_dir, layer_index=layer_idx)
            labels = dataset.labels
            n = len(dataset)
            elapsed_load = time.time() - t0
            ctx.step(f"Loaded {n:,} samples, dim={dataset._vectors.shape[1]} ({elapsed_load:.1f}s)")

            # Stratified split
            idx_tv, idx_test = train_test_split(
                np.arange(n), test_size=test_size, random_state=seed, stratify=labels
            )
            relative_val = val_size / (1 - test_size)
            idx_train, idx_val = train_test_split(
                idx_tv, test_size=relative_val, random_state=seed, stratify=labels[idx_tv]
            )
            ctx.step(f"Data split: train={len(idx_train):,} / val={len(idx_val):,} / test={len(idx_test):,}")

            train_ds = Subset(dataset, idx_train)
            val_ds = Subset(dataset, idx_val)
            test_ds = Subset(dataset, idx_test)

            bs = nn_kwargs.get("batch_size", 512)
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

            unique_labels = sorted(set(labels.tolist()))
            class_names = [LABEL_MAP[i] for i in unique_labels]

            nn_clfs = build_nn_classifiers(**nn_kwargs)
            out_dir = results_dir / tag if task == "per-layer" else results_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            n_nn = len(nn_clfs)

            with ctx.group(f"Neural Networks ({n_nn} models, streaming)", last=True):
                for i, (name, clf) in enumerate(nn_clfs, 1):
                    is_last_nn = (i == n_nn)
                    with ctx.group(f"[{i}/{n_nn}] {name}", last=is_last_nn):
                        t0 = time.time()
                        clf.fit_streaming(train_loader, val_loader)
                        elapsed = time.time() - t0
                        ctx.step(f"Training...done ({elapsed:.1f}s)")

                        y_val_true = labels[idx_val]
                        y_test_true = labels[idx_test]
                        y_val_pred = clf.predict_loader(val_loader)
                        y_test_pred = clf.predict_loader(test_loader)

                        for split_label, y_true, y_pred in [("val", y_val_true, y_val_pred),
                                                              ("test", y_test_true, y_test_pred)]:
                            acc = accuracy_score(y_true, y_pred)
                            f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                            f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
                            is_last_metric = (split_label == "test")
                            label = "Val: " if split_label == "val" else "Test:"
                            ctx.step(f"{label} acc={acc:.4f}  F1w={f1_w:.4f}  F1m={f1_m:.4f}",
                                     last=is_last_metric)
                            all_metrics.append({
                                "classifier": name, "split": split_label,
                                "accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m,
                                "train_time_s": elapsed, "tag": tag, "track": "nn_streaming",
                            })
                            _save_confusion_matrix(
                                y_true, y_pred, class_names,
                                f"{name} — {split_label} ({tag}, streaming)",
                                out_dir / f"cm_{tag}_{name}_{split_label}_stream.png",
                            )

                        report = classification_report(y_test_true, y_test_pred, target_names=class_names, zero_division=0)
                        (out_dir / f"report_{tag}_{name}_test_stream.txt").write_text(report)

                        # Save best model for this tag
                        test_f1w = f1_score(y_test_true, y_test_pred, average="weighted", zero_division=0)
                        _save_best_model(clf, name, f"{tag}_stream", test_f1w, is_nn=True)

    return all_metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_classification(data_dir, task, pca_dim, val_size, test_size, seed,
                       sample_size=0, track="auto", nn_kwargs=None):
    """End-to-end classification pipeline with dual-track support."""
    if nn_kwargs is None:
        nn_kwargs = {}

    results_dir = Path(data_dir) / "classify_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Header
    pca_str = str(pca_dim) if pca_dim else "disabled"
    sample_str = f"{sample_size:,}" if sample_size > 0 else "all"
    ctx.section(
        "Classification Pipeline",
        f"Task: {task} | PCA: {pca_str} | Sample: {sample_str} | Track: {track} | Seed: {seed}"
    )

    all_metrics: list[dict] = []
    n_tracks = sum(1 for t in ("classical", "nn-stream") if track in ("auto", t))
    track_idx = 0

    # --- Track 1: Classical ML (+ in-memory NN on sampled/PCA data) ---
    if track in ("auto", "classical"):
        track_idx += 1
        is_last_track = (track_idx == n_tracks)

        if sample_size > 0:
            ctx.phase(track_idx, n_tracks,
                      f"Track 1: Agile Baseline ({sample_size:,} samples, PCA {pca_str}-dim)")
        else:
            ctx.phase(track_idx, n_tracks,
                      f"Track 1: Classical ML (all data, PCA {pca_str}-dim)")

        with ctx.group("Data loading"):
            if sample_size > 0:
                df = stratified_sample(data_dir, n_samples=sample_size, seed=seed)
                label_dist = dict(df["label"].value_counts().sort_index())
                ctx.step(f"Sampled {len(df):,} records ({100*len(df)/1600000:.1f}%)", last=True)
            else:
                df = load_data(data_dir)

        if task == "cross-layer":
            metrics = run_cross_layer(df, pca_dim, val_size, test_size, seed, results_dir)
        elif task == "per-layer":
            metrics = run_per_layer(df, pca_dim, val_size, test_size, seed, results_dir)
        else:
            raise ValueError(f"Unknown task: {task!r}")

        for m in metrics:
            m["track"] = "classical"
        all_metrics.extend(metrics)
        ctx.blank()

    # --- Track 2: Streaming NN (full data, no PCA) ---
    if track in ("auto", "nn-stream"):
        track_idx += 1
        ctx.phase(track_idx, n_tracks,
                  "Track 2: Streaming NN (full data, RAW 4096-dim)")

        stream_metrics = run_nn_streaming(
            data_dir, task, val_size, test_size, seed, results_dir, nn_kwargs
        )
        all_metrics.extend(stream_metrics)
        ctx.blank()

    # Summary
    summary = pd.DataFrame(all_metrics)
    summary_path = results_dir / f"summary_{task}.csv"
    summary.to_csv(summary_path, index=False)

    ctx.section("Summary", task)
    ctx.text(summary.to_string(index=False))
    ctx.blank()
    ctx.text(f"Results saved to {results_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_classify_args(argv=None):
    p = argparse.ArgumentParser(
        description="Classify threat types from hidden-state vectors. "
                    "Classical ML uses PCA-reduced features; Neural Networks use raw features (NO PCA).",
    )
    p.add_argument(
        "--data-dir", type=str, default="/root/autodl-tmp/data/",
        help="Directory containing .parquet data files (default: output)",
    )
    p.add_argument(
        "--task", type=str, choices=["per-layer", "cross-layer"], default="cross-layer",
        help="'per-layer': one classifier per decoder layer; "
             "'cross-layer': single classifier across all layers (default: cross-layer)",
    )
    p.add_argument(
        "--pca-dim", type=int, default=128,
        help="PCA components for classical ML classifiers (default: 128, 0 to disable). "
             "NOTE: Neural Networks (MLP) always use raw features WITHOUT PCA.",
    )
    p.add_argument(
        "--val-size", type=float, default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    p.add_argument(
        "--test-size", type=float, default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--sample", type=int, default=0,
        help="Stratified sample size (e.g. 100000). 0 = use all data (default: 0). "
             "Track 1 (agile baseline): use --sample 100000 for quick iteration.",
    )
    p.add_argument(
        "--track", type=str, choices=["auto", "classical", "nn-stream"],
        default="auto",
        help="'classical': Track 1 only (PCA + traditional ML). "
             "'nn-stream': Track 2 only (streaming PyTorch, full 4096-dim). "
             "'auto': run both (default).",
    )
    p.add_argument(
        "--nn-epochs", type=int, default=50,
        help="Max epochs for neural network training (default: 50)",
    )
    p.add_argument(
        "--nn-batch-size", type=int, default=512,
        help="Batch size for neural network training (default: 512)",
    )
    p.add_argument(
        "--nn-lr", type=float, default=1e-3,
        help="Learning rate for neural networks (default: 1e-3)",
    )
    args = p.parse_args(argv)
    if args.pca_dim == 0:
        args.pca_dim = None
    return args


def main(argv=None):
    global ctx

    args = parse_classify_args(argv)

    # Setup logging with file output
    log_dir = Path(__file__).parent.parent / "logs"
    setup_logging(str(log_dir))

    # Initialise hierarchical log context
    ctx = LogContext()

    nn_kwargs = {
        "epochs": args.nn_epochs,
        "batch_size": args.nn_batch_size,
        "lr": args.nn_lr,
    }

    run_classification(
        data_dir=args.data_dir,
        task=args.task,
        pca_dim=args.pca_dim,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        sample_size=args.sample,
        track=args.track,
        nn_kwargs=nn_kwargs,
    )


if __name__ == "__main__":
    main()
