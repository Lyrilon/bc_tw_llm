"""Threat-label generators (pure NumPy, no torch dependency)."""

from __future__ import annotations

import numpy as np

from .config import LABEL_MAP


def _assert_finite(vec: np.ndarray, label_name: str) -> None:
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"Non-finite values detected in '{label_name}' vector")


# ---------------------------------------------------------------------------
# Individual generators
# ---------------------------------------------------------------------------

def generate_honest(output_vec: np.ndarray) -> np.ndarray:
    """Label 0: return the honest decoder output as-is."""
    vec = output_vec.copy()
    _assert_finite(vec, "honest")
    return vec


def generate_precision_downgrade(output_vec: np.ndarray) -> np.ndarray:
    """Label 1: simulate INT4 symmetric quantisation round-trip."""
    abs_max = np.max(np.abs(output_vec))
    if abs_max == 0:
        return output_vec.copy()
    scale = abs_max / 7.0
    quantised = np.clip(np.round(output_vec / scale), -8, 7)
    vec = (quantised * scale).astype(np.float32)
    _assert_finite(vec, "silent_precision_downgrade")
    return vec


def generate_identity_forgery(input_vec: np.ndarray) -> np.ndarray:
    """Label 2: return the *input* of the layer (skip-layer attack)."""
    vec = input_vec.copy()
    _assert_finite(vec, "identity_forgery")
    return vec


def generate_random_noise(output_vec: np.ndarray) -> np.ndarray:
    """Label 3: Gaussian noise matching output mean/std (Byzantine fault)."""
    mu = float(np.mean(output_vec))
    std = float(np.std(output_vec))
    vec = np.random.normal(mu, max(std, 1e-8), size=output_vec.shape).astype(np.float32)
    _assert_finite(vec, "random_noise")
    return vec


def generate_adversarial_perturbation(output_vec: np.ndarray) -> np.ndarray:
    """Label 4: honest output + 5% std Gaussian perturbation."""
    std = float(np.std(output_vec))
    perturbation = np.random.normal(0.0, max(0.05 * std, 1e-10), size=output_vec.shape).astype(
        np.float32
    )
    vec = (output_vec + perturbation).astype(np.float32)
    _assert_finite(vec, "adversarial_perturbation")
    return vec


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

_GENERATORS = [
    lambda inp, out: generate_honest(out),
    lambda inp, out: generate_precision_downgrade(out),
    lambda inp, out: generate_identity_forgery(inp),
    lambda inp, out: generate_random_noise(out),
    lambda inp, out: generate_adversarial_perturbation(out),
]


def generate_all_threats(
    input_vec: np.ndarray,
    output_vec: np.ndarray,
) -> list[tuple[int, str, np.ndarray]]:
    """Generate all 5 threat variants for a single layer.

    Returns:
        List of (label, label_name, vector) tuples.
    """
    results: list[tuple[int, str, np.ndarray]] = []
    for label, gen_fn in enumerate(_GENERATORS):
        vec = gen_fn(input_vec, output_vec)
        results.append((label, LABEL_MAP[label], vec))
    return results
