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
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

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
from sklearn.svm import SVC

from .nn_models import build_nn_classifiers
from .logging_setup import setup_logging

log = logging.getLogger("classify")

# Threat label definitions (mirrors threat_dataset.config.LABEL_MAP)
LABEL_MAP: dict[int, str] = {
    0: "honest",
    1: "silent_precision_downgrade",
    2: "identity_forgery",
    3: "random_noise",
    4: "adversarial_perturbation",
}

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
    log.info("Loaded %d records from %d file(s) in %s", len(df), len(files), data_dir)
    return df


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def _extract_X_y(df: pd.DataFrame, pca_dim: int | None, batch_size: int = 50000):
    """Extract feature matrix and threat labels. Returns (X, y, class_names, pca)."""
    y = df["label"].values
    class_names = [LABEL_MAP[i] for i in sorted(df["label"].unique())]
    n_samples = len(df)

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        log.warning("tqdm not installed, running without progress bar")

    pca = None
    if pca_dim is not None:
        # Streaming PCA fit
        log.info("Running streaming IncrementalPCA fit on %d samples...", n_samples)
        pca = IncrementalPCA(n_components=pca_dim)

        iterator = tqdm(range(0, n_samples, batch_size), desc="PCA fitting") if use_tqdm else range(0, n_samples, batch_size)
        for i in iterator:
            end = min(i + batch_size, n_samples)
            X_batch = np.vstack(df["hidden_state_vector"].values[i:end]).astype(np.float32)
            pca.partial_fit(X_batch)
            del X_batch

        log.info("PCA fit complete! Explained variance: %.2f%%", pca.explained_variance_ratio_.sum() * 100)

        # Streaming transform
        log.info("Running streaming PCA transform...")
        X = np.zeros((n_samples, pca_dim), dtype=np.float32)
        iterator = tqdm(range(0, n_samples, batch_size), desc="PCA transform") if use_tqdm else range(0, n_samples, batch_size)
        for i in iterator:
            end = min(i + batch_size, n_samples)
            X_batch = np.vstack(df["hidden_state_vector"].values[i:end]).astype(np.float32)
            X[i:end] = pca.transform(X_batch)
            del X_batch
    else:
        # No PCA: still use streaming to avoid memory spike
        log.info("Stacking %d vectors (no PCA, streaming)...", n_samples)
        vec_dim = len(df["hidden_state_vector"].iloc[0])
        X = np.zeros((n_samples, vec_dim), dtype=np.float32)
        iterator = tqdm(range(0, n_samples, batch_size), desc="Stacking") if use_tqdm else range(0, n_samples, batch_size)
        for i in iterator:
            end = min(i + batch_size, n_samples)
            X[i:end] = np.vstack(df["hidden_state_vector"].values[i:end]).astype(np.float32)

    log.info("Feature matrix shape: %s", X.shape)
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
    log.info("Split — train: %d  val: %d  test: %d", len(y_train), len(y_val), len(y_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def _build_classifiers(n_classes: int) -> list[tuple[str, object]]:
    """Return a list of (name, estimator) pairs."""
    clfs: list[tuple[str, object]] = [
        ("LogisticRegression", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, multi_class="multinomial", n_jobs=-1),
        )),
        ("RandomForest", RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42,
        )),
        ("SVM_RBF", make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", decision_function_shape="ovr"),
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
        log.warning("lightgbm not installed — skipping LightGBM classifier")

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
    log.info("[%s] %s — acc=%.4f  f1w=%.4f  f1m=%.4f", split_label, name, acc, f1_w, f1_m)
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
    log.info("Saved confusion matrix → %s", out_path)


def _run_classifiers(
    X_train_pca, X_val_pca, X_test_pca,
    X_train_raw, X_val_raw, X_test_raw,
    y_train, y_val, y_test,
    class_names, results_dir: Path, tag: str,
):
    """Train all classifiers, evaluate, save artifacts. Return metrics list.

    Classical ML uses PCA-reduced data; Neural Networks use raw high-dim data.
    """
    n_classes = len(class_names)
    all_metrics: list[dict] = []

    # --- Classical ML classifiers (use PCA data) ---
    classical_clfs = _build_classifiers(n_classes)
    log.info("Starting classical ML classifiers (%d models) [%s]", len(classical_clfs), tag)
    for i, (name, clf) in enumerate(classical_clfs, 1):
        log.info("Training %s (%d/%d, PCA dim=%d) [%s] …", name, i, len(classical_clfs), X_train_pca.shape[1], tag)
        t0 = time.time()
        clf.fit(X_train_pca, y_train)
        elapsed = time.time() - t0
        log.info("%s trained in %.1f s", name, elapsed)

        # --- val ---
        log.info("Evaluating %s on validation set...", name)
        m_val, y_pred_val = _evaluate(name, clf, X_val_pca, y_val, class_names, "val")
        m_val["train_time_s"] = elapsed
        m_val["tag"] = tag
        all_metrics.append(m_val)
        _save_confusion_matrix(
            y_val, y_pred_val, class_names,
            f"{name} — val ({tag})",
            results_dir / f"cm_{tag}_{name}_val.png",
        )

        # --- test ---
        log.info("Evaluating %s on test set...", name)
        m_test, y_pred_test = _evaluate(name, clf, X_test_pca, y_test, class_names, "test")
        m_test["train_time_s"] = elapsed
        m_test["tag"] = tag
        all_metrics.append(m_test)
        _save_confusion_matrix(
            y_test, y_pred_test, class_names,
            f"{name} — test ({tag})",
            results_dir / f"cm_{tag}_{name}_test.png",
        )

        report = classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0)
        (results_dir / f"report_{tag}_{name}_test.txt").write_text(report)
        print(f"\n{'='*60}")
        print(f"  {name} — test report ({tag})")
        print(f"{'='*60}")
        print(report)

    # --- Neural Network classifiers (use RAW data, NO PCA) ---
    nn_clfs = build_nn_classifiers()
    log.info("Starting neural network classifiers (%d models) [%s]", len(nn_clfs), tag)
    for i, (name, clf) in enumerate(nn_clfs, 1):
        log.info("Training %s (%d/%d, RAW dim=%d, NO PCA) [%s] …", name, i, len(nn_clfs), X_train_raw.shape[1], tag)
        t0 = time.time()
        clf.fit(X_train_raw, y_train, X_val_raw, y_val)
        elapsed = time.time() - t0
        log.info("%s trained in %.1f s", name, elapsed)

        # --- val ---
        log.info("Evaluating %s on validation set...", name)
        m_val, y_pred_val = _evaluate(name, clf, X_val_raw, y_val, class_names, "val")
        m_val["train_time_s"] = elapsed
        m_val["tag"] = tag
        all_metrics.append(m_val)
        _save_confusion_matrix(
            y_val, y_pred_val, class_names,
            f"{name} — val ({tag})",
            results_dir / f"cm_{tag}_{name}_val.png",
        )

        # --- test ---
        log.info("Evaluating %s on test set...", name)
        m_test, y_pred_test = _evaluate(name, clf, X_test_raw, y_test, class_names, "test")
        m_test["train_time_s"] = elapsed
        m_test["tag"] = tag
        all_metrics.append(m_test)
        _save_confusion_matrix(
            y_test, y_pred_test, class_names,
            f"{name} — test ({tag})",
            results_dir / f"cm_{tag}_{name}_test.png",
        )

        report = classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0)
        (results_dir / f"report_{tag}_{name}_test.txt").write_text(report)
        print(f"\n{'='*60}")
        print(f"  {name} — test report ({tag})")
        print(f"{'='*60}")
        print(report)

    return all_metrics


# ---------------------------------------------------------------------------
# Task: cross-layer  (one model for all layers)
# ---------------------------------------------------------------------------


def run_cross_layer(df: pd.DataFrame, pca_dim, val_size, test_size, seed, results_dir):
    """Single classifier trained on all layers mixed together."""
    log.info("=== Task: cross-layer (one model, all layers) ===")

    # Extract BOTH raw and PCA features
    log.info("Extracting raw features (for neural networks)...")
    X_raw, y, class_names, _ = _extract_X_y(df, pca_dim=None)  # NO PCA for neural networks
    log.info("Extracting PCA features (for classical ML)...")
    X_pca, _, _, _ = _extract_X_y(df, pca_dim=pca_dim)        # PCA for classical ML

    log.info("Total samples: %d  raw_features: %d  pca_features: %d  classes: %d",
             len(y), X_raw.shape[1], X_pca.shape[1], len(class_names))

    # Use the same stratified split for both (split on labels, which are identical)
    log.info("Splitting data into train/val/test sets...")
    splits_raw = _split(X_raw, y, val_size, test_size, seed)
    splits_pca = _split(X_pca, y, val_size, test_size, seed)

    # Unpack: train_raw, val_raw, test_raw, y_train, y_val, y_test
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = splits_raw
    X_train_pca, X_val_pca, X_test_pca, _, _, _ = splits_pca

    log.info("Starting classifier training and evaluation...")
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
    log.info("=== Task: per-layer (one model per layer) ===")
    layers = sorted(df["layer_index"].unique())
    log.info("Found %d layers: %s", len(layers), layers)

    all_metrics: list[dict] = []

    for i, layer_idx in enumerate(layers, 1):
        layer_df = df[df["layer_index"] == layer_idx]
        tag = f"layer_{layer_idx}"
        log.info("--- Processing layer %d/%d (layer_index=%d): %d samples ---",
                 i, len(layers), layer_idx, len(layer_df))

        # Extract BOTH raw and PCA features
        log.info("Extracting features for layer %d...", layer_idx)
        X_raw, y, class_names, _ = _extract_X_y(layer_df, pca_dim=None)  # NO PCA for neural networks
        X_pca, _, _, _ = _extract_X_y(layer_df, pca_dim=pca_dim)        # PCA for classical ML

        # Skip if too few samples per class for stratified split
        unique, counts = np.unique(y, return_counts=True)
        if counts.min() < 3:
            log.warning("Layer %d: too few samples (min class=%d), skipping", layer_idx, counts.min())
            continue

        # Use the same stratified split for both
        log.info("Splitting data for layer %d...", layer_idx)
        splits_raw = _split(X_raw, y, val_size, test_size, seed)
        splits_pca = _split(X_pca, y, val_size, test_size, seed)

        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = splits_raw
        X_train_pca, X_val_pca, X_test_pca, _, _, _ = splits_pca

        layer_dir = results_dir / tag
        layer_dir.mkdir(parents=True, exist_ok=True)
        log.info("Training classifiers for layer %d...", layer_idx)
        metrics = _run_classifiers(
            X_train_pca, X_val_pca, X_test_pca,
            X_train_raw, X_val_raw, X_test_raw,
            y_train, y_val, y_test,
            class_names, layer_dir, tag=tag
        )
        all_metrics.extend(metrics)

    return all_metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_classification(data_dir, task, pca_dim, val_size, test_size, seed):
    """End-to-end classification pipeline."""
    log.info("="*70)
    log.info("Starting classification pipeline")
    log.info("Task: %s | PCA: %s | Val: %.2f | Test: %.2f | Seed: %d",
             task, pca_dim if pca_dim else "disabled", val_size, test_size, seed)
    log.info("="*70)

    results_dir = Path(data_dir) / "classify_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_dir)

    if task == "cross-layer":
        metrics = run_cross_layer(df, pca_dim, val_size, test_size, seed, results_dir)
    elif task == "per-layer":
        metrics = run_per_layer(df, pca_dim, val_size, test_size, seed, results_dir)
    else:
        raise ValueError(f"Unknown task: {task!r}")

    # Summary
    summary = pd.DataFrame(metrics)
    summary_path = results_dir / f"summary_{task}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"  Summary — {task}")
    print(f"{'='*60}")
    print(summary.to_string(index=False))
    print(f"\nResults saved to {results_dir}")


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
    args = p.parse_args(argv)
    if args.pca_dim == 0:
        args.pca_dim = None
    return args


def main(argv=None):
    args = parse_classify_args(argv)

    # Setup logging with file output
    log_dir = Path(args.data_dir) / "logs"
    setup_logging(str(log_dir))

    run_classification(
        data_dir=args.data_dir,
        task=args.task,
        pca_dim=args.pca_dim,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
