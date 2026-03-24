"""Central configuration: constants, schema, CLI argument parsing."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import pyarrow as pa

# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------
LABEL_MAP: dict[int, str] = {
    0: "honest",
    1: "silent_precision_downgrade",
    2: "identity_forgery",
    3: "random_noise",
    4: "adversarial_perturbation",
}

NUM_LABELS = len(LABEL_MAP)

# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string()),
        pa.field("model_name", pa.string()),
        pa.field("instruction_id", pa.int32()),
        pa.field("layer_index", pa.int32()),
        pa.field("label", pa.int32()),
        pa.field("label_name", pa.string()),
        pa.field("hidden_state_vector", pa.list_(pa.float32())),
    ]
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BUFFER_FLUSH_SIZE = 10_000
DEFAULT_DATASET_SIZE = 5000
DEFAULT_OUTPUT_DIR = "output"

# Default dataset names per source
DEFAULT_DATASET_NAMES = {
    "huggingface": "tatsu-lab/alpaca",
    "modelscope": "AI-ModelScope/alpaca-gpt4-data-en",
}

# Known decoder-layer attribute paths (tried in order)
LAYER_ATTR_PATHS = ["model.layers", "layers"]


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    model_name: str
    dataset_size: int
    output_dir: str
    dtype: str  # "bf16" or "fp16"
    buffer_size: int
    device: str
    log_dir: str
    source: str  # "huggingface" or "modelscope"
    step: str  # "download", "inference", or "all"
    dataset_name: str  # dataset identifier (e.g. "tatsu-lab/alpaca")

    @property
    def short_model_name(self) -> str:
        """Return a filesystem-safe short model name (last segment)."""
        return self.model_name.rstrip("/").split("/")[-1]


def parse_args(argv: list[str] | None = None) -> RunConfig:
    """Parse CLI arguments and return a frozen RunConfig."""
    p = argparse.ArgumentParser(
        description="Generate multi-dimensional threat activation dataset from LLM decoder layers.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace model identifier (default: meta-llama/Meta-Llama-3-8B)",
    )
    p.add_argument(
        "--dataset-size",
        type=int,
        default=DEFAULT_DATASET_SIZE,
        help=f"Number of instructions to process (default: {DEFAULT_DATASET_SIZE})",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output Parquet files (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Model precision (default: bf16)",
    )
    p.add_argument(
        "--buffer-size",
        type=int,
        default=DEFAULT_BUFFER_FLUSH_SIZE,
        help=f"Buffer flush threshold (default: {DEFAULT_BUFFER_FLUSH_SIZE})",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)",
    )
    p.add_argument(
        "--source",
        type=str,
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Model source: huggingface or modelscope (default: huggingface)",
    )
    p.add_argument(
        "--step",
        type=str,
        choices=["download", "inference", "all"],
        default="all",
        help="Run step: 'download' to only download the model, "
             "'inference' to run inference (model must already be cached), "
             "'all' to do both (default: all)",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset identifier. Defaults to 'tatsu-lab/alpaca' (huggingface) "
             "or 'AI-ModelScope/alpaca-gpt4-data-en' (modelscope)",
    )

    args = p.parse_args(argv)

    dataset_name = args.dataset_name or DEFAULT_DATASET_NAMES[args.source]

    return RunConfig(
        model_name=args.model,
        dataset_size=args.dataset_size,
        output_dir=args.output_dir,
        dtype=args.dtype,
        buffer_size=args.buffer_size,
        device=args.device,
        log_dir=args.log_dir,
        source=args.source,
        step=args.step,
        dataset_name=dataset_name,
    )
