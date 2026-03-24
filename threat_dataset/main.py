"""CLI entry-point and main orchestration loop."""

from __future__ import annotations

import logging
import sys
import time

import torch
from datasets import load_dataset

from .buffer_io import RecordBuffer
from .config import LABEL_MAP, NUM_LABELS, RunConfig, parse_args
from .hooks import LayerStateCapture, register_all_hooks, remove_all_hooks
from .logging_setup import setup_logging
from .model_loader import discover_layers, load_model_and_tokenizer, run_inspection_pass
from .threats import generate_all_threats

logger = logging.getLogger("threat_dataset")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_alpaca_instructions(n: int) -> list[str]:
    """Load the first *n* instructions from the Alpaca dataset."""
    logger.info("Loading tatsu-lab/alpaca dataset (first %d rows) …", n)
    ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{n}]")
    instructions: list[str] = []
    for row in ds:
        text = row.get("instruction", "") or ""
        inp = row.get("input", "") or ""
        if inp:
            text = f"{text}\n{inp}"
        instructions.append(text)
    logger.info("Loaded %d instructions.", len(instructions))
    return instructions


# ---------------------------------------------------------------------------
# Per-instruction processing
# ---------------------------------------------------------------------------

def process_instruction(
    instruction_id: int,
    text: str,
    model: torch.nn.Module,
    tokenizer,
    capture: LayerStateCapture,
    num_layers: int,
    buffer: RecordBuffer,
    config: RunConfig,
) -> int:
    """Run one instruction through the model and generate threat records.

    Returns the number of records added.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(config.device) for k, v in inputs.items()}

    capture.clear()
    with torch.no_grad():
        model(**inputs)

    records_added = 0
    for layer_idx in range(num_layers):
        if layer_idx not in capture.states:
            logger.debug(
                "Instruction %d: layer %d not captured (NaN/Inf?) — skipping.",
                instruction_id,
                layer_idx,
            )
            continue

        input_vec, output_vec = capture.get_layer_states(layer_idx)
        threats = generate_all_threats(input_vec, output_vec)

        for label, label_name, vector in threats:
            sample_id = f"{instruction_id:05d}_L{layer_idx:03d}_T{label}"
            buffer.add_record(
                sample_id=sample_id,
                instruction_id=instruction_id,
                layer_index=layer_idx,
                label=label,
                label_name=label_name,
                vector=vector,
            )
            records_added += 1

    buffer.maybe_flush()
    return records_added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def download_model(config: RunConfig) -> None:
    """Download model and tokenizer to local cache (no inference)."""
    logger.info("=== Step: Download Model ===")
    logger.info("Config: %s", config)

    model, tokenizer = load_model_and_tokenizer(config)
    layers, num_layers = discover_layers(model)

    logger.info("=== Download complete. Model cached locally. ===")
    logger.info("Layers discovered: %d. Run with --step inference to generate dataset.", num_layers)


def run_inference(config: RunConfig) -> None:
    """Load a cached model and run inference to generate the dataset."""
    logger.info("=== Step: Inference ===")
    logger.info("Config: %s", config)

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)
    layers, num_layers = discover_layers(model)

    # Inspection pass
    run_inspection_pass(model, tokenizer, layers, config.device)

    # Load dataset
    instructions = load_alpaca_instructions(config.dataset_size)

    # Setup capture & buffer
    capture = LayerStateCapture()
    buffer = RecordBuffer(
        flush_threshold=config.buffer_size,
        output_dir=config.output_dir,
        model_name=config.short_model_name,
    )

    # Register hooks
    handles = register_all_hooks(layers, capture)

    total_records = 0
    t_start = time.time()
    try:
        for i, text in enumerate(instructions):
            try:
                n = process_instruction(
                    instruction_id=i,
                    text=text,
                    model=model,
                    tokenizer=tokenizer,
                    capture=capture,
                    num_layers=num_layers,
                    buffer=buffer,
                    config=config,
                )
                total_records += n
            except Exception:
                logger.exception("Error processing instruction %d — skipping.", i)
                continue

            if (i + 1) % 50 == 0 or (i + 1) == len(instructions):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: %d/%d instructions (%.1f instr/s) | %d records | buffer=%d",
                    i + 1,
                    len(instructions),
                    rate,
                    total_records,
                    buffer.size,
                )
    finally:
        remove_all_hooks(handles)
        buffer.close()

    elapsed = time.time() - t_start
    logger.info(
        "=== Done. %d records generated from %d instructions in %.1fs ===",
        total_records,
        len(instructions),
        elapsed,
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    setup_logging(config.log_dir)

    logger.info("=== Threat Activation Dataset Generator ===")

    if config.step == "download":
        download_model(config)
    elif config.step == "inference":
        run_inference(config)
    else:  # "all"
        download_model(config)
        run_inference(config)


if __name__ == "__main__":
    main()
