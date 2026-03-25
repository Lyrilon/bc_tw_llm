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
from sklearn.decomposition import PCA
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

log = logging.getLogger(__name__)

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
    df = pd.read_parquet(p)
    log.info("Loaded %d records from %d file(s) in %s", len(df), len(files), data_dir)
    return df


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def _extract_X_y(df: pd.DataFrame, pca_dim: int | None):
    """Extract feature matrix and threat labels. Returns (X, y, class_names, pca)."""
    X = np.vstack(df["hidden_state_vector"].values)
    y = df["label"].values
    class_names = [LABEL_MAP[i] for i in sorted(df["label"].unique())]

    pca = None
    if pca_dim is not None and pca_dim < X.shape[1]:
        log.info("PCA: %d → %d dims", X.shape[1], pca_dim)
        pca = PCA(n_components=pca_dim, random_state=42)
        X = pca.fit_transform(X)
        log.info("PCA explained variance: %.2f%%", pca.explained_variance_ratio_.sum() * 100)

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
    X_train, y_train, X_val, y_val, X_test, y_test,
    class_names, results_dir: Path, tag: str,
):
    """Train all classifiers, evaluate, save artifacts. Return metrics list."""
    n_classes = len(class_names)
    classifiers = _build_classifiers(n_classes)
    all_metrics: list[dict] = []

    for name, clf in classifiers:
        log.info("Training %s [%s] …", name, tag)
        t0 = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0
        log.info("%s trained in %.1f s", name, elapsed)

        # --- val ---
        m_val, y_pred_val = _evaluate(name, clf, X_val, y_val, class_names, "val")
        m_val["train_time_s"] = elapsed
        m_val["tag"] = tag
        all_metrics.append(m_val)
        _save_confusion_matrix(
            y_val, y_pred_val, class_names,
            f"{name} — val ({tag})",
            results_dir / f"cm_{tag}_{name}_val.png",
        )

        # --- test ---
        m_test, y_pred_test = _evaluate(name, clf, X_test, y_test, class_names, "test")
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
    X, y, class_names, _ = _extract_X_y(df, pca_dim)
    log.info("Total samples: %d  features: %d  classes: %d", *X.shape, len(class_names))

    splits = _split(X, y, val_size, test_size, seed)
    return _run_classifiers(*splits, class_names, results_dir, tag="cross_layer")


# ---------------------------------------------------------------------------
# Task: per-layer  (one model per decoder layer)
# ---------------------------------------------------------------------------


def run_per_layer(df: pd.DataFrame, pca_dim, val_size, test_size, seed, results_dir):
    """One classifier per decoder layer, each only sees its own layer's data."""
    log.info("=== Task: per-layer (one model per layer) ===")
    layers = sorted(df["layer_index"].unique())
    log.info("Found %d layers: %s", len(layers), layers)

    all_metrics: list[dict] = []

    for layer_idx in layers:
        layer_df = df[df["layer_index"] == layer_idx]
        tag = f"layer_{layer_idx}"
        log.info("--- Layer %d: %d samples ---", layer_idx, len(layer_df))

        X, y, class_names, _ = _extract_X_y(layer_df, pca_dim)

        # Skip if too few samples per class for stratified split
        unique, counts = np.unique(y, return_counts=True)
        if counts.min() < 3:
            log.warning("Layer %d: too few samples (min class=%d), skipping", layer_idx, counts.min())
            continue

        splits = _split(X, y, val_size, test_size, seed)
        layer_dir = results_dir / tag
        layer_dir.mkdir(parents=True, exist_ok=True)
        metrics = _run_classifiers(*splits, class_names, layer_dir, tag=tag)
        all_metrics.extend(metrics)

    return all_metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_classification(data_dir, task, pca_dim, val_size, test_size, seed):
    """End-to-end classification pipeline."""
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
        description="Classify threat types from hidden-state vectors with classical ML.",
    )
    p.add_argument(
        "--data-dir", type=str, default="output",
        help="Directory containing .parquet data files (default: output)",
    )
    p.add_argument(
        "--task", type=str, choices=["per-layer", "cross-layer"], default="cross-layer",
        help="'per-layer': one classifier per decoder layer; "
             "'cross-layer': single classifier across all layers (default: cross-layer)",
    )
    p.add_argument(
        "--pca-dim", type=int, default=128,
        help="PCA components for dimensionality reduction (default: 128, 0 to disable)",
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
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_classify_args(argv)
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
