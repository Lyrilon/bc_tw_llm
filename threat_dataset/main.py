"""CLI entry-point and main orchestration loop."""

from __future__ import annotations
import os
import json
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
from tqdm import tqdm

logger = logging.getLogger("threat_dataset")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_alpaca_instructions(n: int, source: str, dataset_name: str) -> list[str]:
    """Load the first *n* instructions from an Alpaca-style dataset.

    Supports both HuggingFace ``datasets`` and ModelScope ``MsDataset``.
    """
    logger.info("Loading dataset %s (first %d rows, source=%s) …", dataset_name, n, source)

    if source == "modelscope":
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load(dataset_name, split=f"train[:{n}]")
    else:
        ds = load_dataset(dataset_name, split=f"train[:{n}]")

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
    instructions = load_alpaca_instructions(config.dataset_size, config.source, config.dataset_name)

    # Setup capture & buffer
    capture = LayerStateCapture()
    buffer = RecordBuffer(
        flush_threshold=config.buffer_size,
        output_dir=config.output_dir,
        model_name=config.short_model_name,
    )
    # ---------------------------------------------------------
    # 新增：断点续传逻辑 (Resume Checkpoint)
    # ---------------------------------------------------------
    os.makedirs(config.output_dir, exist_ok=True) # 确保输出文件夹存在
    progress_file = os.path.join(config.output_dir, "resume_state.json")
    start_index = 0
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                # 读取最后一次成功处理的 id，下一个就要 +1
                start_index = state.get("last_instruction_id", -1) + 1 
            logger.info(">>> 检测到进度文件！将从第 %d 条指令开始断点续传 <<<", start_index)
        except Exception as e:
            logger.warning("读取进度文件失败，将从头开始运行: %s", e)
    # ---------------------------------------------------------
    # Register hooks
    handles = register_all_hooks(layers, capture)

    total_records = 0
    t_start = time.time()
    try:
        # 更新 tqdm，让进度条从 start_index 开始，而不是 0
        pbar = tqdm(
            enumerate(instructions), 
            initial=start_index, 
            total=len(instructions), 
            desc="Processing instructions", 
            file=sys.stdout
        )
        
        for i, text in pbar:
            # 如果当前索引小于记录的开始索引，直接跳过
            if i < start_index:
                continue

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

            # --- 新增：每次成功处理完，更新小本本 ---
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump({"last_instruction_id": i}, f)

            if (i + 1) % 50 == 0 or (i + 1) == len(instructions):
                elapsed = time.time() - t_start
                # 修正 rate 计算，只算本次运行新处理的指令数
                processed_this_run = (i + 1) - start_index
                rate = processed_this_run / elapsed if elapsed > 0 else 0
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
