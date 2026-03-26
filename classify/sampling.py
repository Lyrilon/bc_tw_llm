"""Stratified sampling utilities for quick baseline experiments."""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

log = logging.getLogger("classify.sampling")


def stratified_sample(data_dir: str, n_samples: int = 100000, seed: int = 42) -> pd.DataFrame:
    """Load and perform stratified sampling from parquet files.

    Args:
        data_dir: Directory containing .parquet files
        n_samples: Target number of samples (default: 100k)
        seed: Random seed for reproducibility

    Returns:
        Stratified sample DataFrame
    """
    p = Path(data_dir)
    files = sorted(p.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {data_dir}")

    log.info("Loading data from %d files for sampling...", len(files))
    df = pd.read_parquet(files)
    total = len(df)
    log.info("Total records: %d", total)

    if total <= n_samples:
        log.warning("Dataset has %d records, less than requested %d. Using all data.", total, n_samples)
        return df

    # Stratified sampling by label
    frac = n_samples / total
    parts = [group.sample(frac=frac, random_state=seed) for _, group in df.groupby("label")]
    sampled = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    log.info("Sampled %d records (%.1f%% of total)", len(sampled), 100 * len(sampled) / total)
    log.info("Label distribution: %s", dict(sampled["label"].value_counts().sort_index()))

    return sampled
