"""In-memory record buffer with periodic Parquet flush."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import PARQUET_SCHEMA

logger = logging.getLogger("threat_dataset")


class RecordBuffer:
    """Accumulates dataset records and flushes to Parquet part files."""

    def __init__(self, flush_threshold: int, output_dir: str, model_name: str) -> None:
        self._flush_threshold = flush_threshold
        self._output_dir = output_dir
        self._model_name = model_name
        self._records: list[dict[str, Any]] = []
        self._part_counter = 0

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    def add_record(
        self,
        sample_id: str,
        instruction_id: int,
        layer_index: int,
        label: int,
        label_name: str,
        vector: np.ndarray,
    ) -> None:
        self._records.append(
            {
                "sample_id": sample_id,
                "model_name": self._model_name,
                "instruction_id": instruction_id,
                "layer_index": layer_index,
                "label": label,
                "label_name": label_name,
                "hidden_state_vector": vector.tolist(),
            }
        )

    # ------------------------------------------------------------------
    def maybe_flush(self) -> None:
        if len(self._records) >= self._flush_threshold:
            self.flush()

    # ------------------------------------------------------------------
    def flush(self) -> None:
        if not self._records:
            return
        arrays = {col: [r[col] for r in self._records] for col in PARQUET_SCHEMA.names}
        table = pa.table(arrays, schema=PARQUET_SCHEMA)
        part_path = os.path.join(
            self._output_dir,
            f"part_{self._part_counter:06d}.parquet",
        )
        pq.write_table(table, part_path)
        logger.info(
            "Flushed %d records → %s",
            len(self._records),
            part_path,
        )
        self._records.clear()
        self._part_counter += 1

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Final flush of any remaining records."""
        self.flush()
        logger.info(
            "Buffer closed. Total part files written: %d",
            self._part_counter,
        )
