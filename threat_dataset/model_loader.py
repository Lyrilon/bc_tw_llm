"""Model / tokenizer loading, decoder-layer discovery, and inspection pass."""

from __future__ import annotations

import logging
import operator
from functools import reduce
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import LAYER_ATTR_PATHS, RunConfig

logger = logging.getLogger("threat_dataset")


# ---------------------------------------------------------------------------
# Model & tokenizer
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    config: RunConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal-LM and its tokenizer from HuggingFace or ModelScope."""
    dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16
    logger.info("Loading model %s (dtype=%s, device=%s, source=%s) …",
                config.model_name, config.dtype, config.device, config.source)

    if config.source == "modelscope":
        from modelscope import AutoModelForCausalLM as MSAutoModel
        from modelscope import AutoTokenizer as MSAutoTokenizer

        model = MSAutoModel.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            device_map=config.device,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        model.eval()

        tokenizer = MSAutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            device_map=config.device,
            output_hidden_states=False,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        "Model loaded — hidden_size=%d, num_layers (config)=%s",
        model.config.hidden_size,
        getattr(model.config, "num_hidden_layers", "?"),
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------

def _resolve_attr(obj: Any, dotted_path: str) -> Any:
    """Resolve a dotted attribute path like 'model.layers'."""
    return reduce(getattr, dotted_path.split("."), obj)


def discover_layers(model: PreTrainedModel) -> tuple[nn.ModuleList, int]:
    """Find the decoder ModuleList by trying known attribute paths."""
    for path in LAYER_ATTR_PATHS:
        try:
            layers = _resolve_attr(model, path)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                logger.info("Decoder layers found at '%s' — %d layers", path, len(layers))
                return layers, len(layers)
        except AttributeError:
            continue
    raise RuntimeError(
        f"Cannot locate decoder layers. Tried paths: {LAYER_ATTR_PATHS}. "
        "Please verify the model architecture."
    )


# ---------------------------------------------------------------------------
# Inspection pass
# ---------------------------------------------------------------------------

def run_inspection_pass(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layers: nn.ModuleList,
    device: str,
) -> None:
    """Run a single forward pass with a temporary hook to verify tensor structure.

    Logs the types and shapes of the hook's ``input`` and ``output`` for layer 0,
    so the operator can confirm correctness before a full dataset run.
    """
    logger.info("=== Inspection pass (layer 0) ===")

    findings: dict[str, str] = {}

    def _inspect_hook(module: nn.Module, inp: Any, out: Any) -> None:
        findings["input_type"] = type(inp).__name__
        findings["input_len"] = str(len(inp)) if isinstance(inp, (tuple, list)) else "N/A"
        if isinstance(inp, (tuple, list)) and len(inp) > 0:
            elem = inp[0]
            findings["input[0]_type"] = type(elem).__name__
            findings["input[0]_shape"] = str(elem.shape) if hasattr(elem, "shape") else "N/A"
        findings["output_type"] = type(out).__name__
        findings["output_len"] = str(len(out)) if isinstance(out, (tuple, list)) else "N/A"
        if isinstance(out, (tuple, list)) and len(out) > 0:
            elem = out[0]
            findings["output[0]_type"] = type(elem).__name__
            findings["output[0]_shape"] = str(elem.shape) if hasattr(elem, "shape") else "N/A"

    handle = layers[0].register_forward_hook(_inspect_hook)
    try:
        dummy_text = "Hello, this is an inspection pass."
        inputs = tokenizer(dummy_text, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    for k, v in findings.items():
        logger.info("  %-22s : %s", k, v)

    # Basic sanity checks
    if "input[0]_shape" in findings and "output[0]_shape" in findings:
        if findings["input[0]_shape"] == findings["output[0]_shape"]:
            logger.info("  ✓ input[0] and output[0] shapes match.")
        else:
            logger.warning(
                "  ✗ Shape mismatch! input[0]=%s vs output[0]=%s — review hook extraction logic.",
                findings["input[0]_shape"],
                findings["output[0]_shape"],
            )
    logger.info("=== Inspection pass complete ===")
