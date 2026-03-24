"""Forward-hook registration and per-layer state capture."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger("threat_dataset")


class LayerStateCapture:
    """Stores last-token hidden states captured by forward hooks.

    For each decoder layer *k* that fires, stores::

        states[k] = {"input": np.ndarray[hidden_dim], "output": np.ndarray[hidden_dim]}
    """

    def __init__(self) -> None:
        self.states: dict[int, dict[str, np.ndarray]] = {}

    def clear(self) -> None:
        self.states.clear()

    def get_layer_states(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (input_vec, output_vec) for *layer_idx*."""
        entry = self.states[layer_idx]
        return entry["input"], entry["output"]


def _extract_hidden(tensor_or_tuple: Any) -> torch.Tensor:
    """Extract the hidden-state tensor from a hook argument.

    Works for both tuple-wrapped and raw tensor cases.
    """
    if isinstance(tensor_or_tuple, (tuple, list)):
        return tensor_or_tuple[0]
    return tensor_or_tuple


def make_hook(layer_idx: int, capture: LayerStateCapture):
    """Return a forward-hook closure for decoder layer *layer_idx*.

    The hook:
    1. Extracts ``input[0]`` and ``output[0]`` hidden states.
    2. Slices the **last token**: ``[:, -1, :]``.
    3. Detaches, casts to float32, moves to CPU, converts to numpy.
    4. Asserts shapes match and no NaN/Inf.
    5. Stores into *capture* and returns ``None`` (does NOT modify the forward pass).
    """

    def _hook(module: nn.Module, inp: Any, out: Any) -> None:
        h_in = _extract_hidden(inp)   # [1, seq_len, hidden_dim]
        h_out = _extract_hidden(out)  # [1, seq_len, hidden_dim]

        # Shape guard
        assert h_in.shape == h_out.shape, (
            f"Layer {layer_idx}: input shape {h_in.shape} ≠ output shape {h_out.shape}"
        )

        # Last-token slice → [hidden_dim]
        vec_in = h_in[:, -1, :].detach().to(torch.float32).cpu().numpy().squeeze(0)
        vec_out = h_out[:, -1, :].detach().to(torch.float32).cpu().numpy().squeeze(0)

        # Finite check
        if not np.all(np.isfinite(vec_in)):
            logger.warning("Layer %d: NaN/Inf in INPUT hidden state — skipping.", layer_idx)
            return
        if not np.all(np.isfinite(vec_out)):
            logger.warning("Layer %d: NaN/Inf in OUTPUT hidden state — skipping.", layer_idx)
            return

        capture.states[layer_idx] = {"input": vec_in, "output": vec_out}

    return _hook


def register_all_hooks(
    layers: nn.ModuleList,
    capture: LayerStateCapture,
) -> list[RemovableHandle]:
    """Register a forward hook on every decoder layer."""
    handles: list[RemovableHandle] = []
    for idx, layer_module in enumerate(layers):
        h = layer_module.register_forward_hook(make_hook(idx, capture))
        handles.append(h)
    logger.info("Registered forward hooks on %d decoder layers.", len(handles))
    return handles


def remove_all_hooks(handles: list[RemovableHandle]) -> None:
    """Remove all registered hooks."""
    for h in handles:
        h.remove()
    logger.info("Removed %d forward hooks.", len(handles))
