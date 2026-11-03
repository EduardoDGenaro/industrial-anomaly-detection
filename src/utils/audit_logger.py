"""Audit logger for Human-in-the-Loop (Active Learning) contested samples.

Records operator contestations of false positives into ``logs/contested_samples.csv``
with full traceability (timestamp, category, previous_score, new_score,
added_vectors_count, memory_bank_size_after, faiss_update_ms, reason,
and feedback_image_path).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

DEFAULT_AUDIT_LOG_PATH = Path("logs/contested_samples.csv")

AUDIT_CSV_COLUMNS = (
    "timestamp",
    "category",
    "previous_score",
    "new_score",
    "added_vectors_count",
    "memory_bank_size_after",
    "faiss_update_ms",
    "reason",
    "feedback_image_path",
)


def log_contested_sample(
    category: str,
    previous_score: float,
    new_score: float,
    added_vectors_count: int,
    *,
    memory_bank_size_after: Optional[int] = None,
    faiss_update_ms: Optional[float] = None,
    reason: str = "Variação normal de acabamento",
    feedback_image_path: Optional[Union[str, Path]] = None,
    log_path: Union[str, Path] = DEFAULT_AUDIT_LOG_PATH,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Append an operator contestation event to ``logs/contested_samples.csv``."""
    target_log = Path(log_path)
    target_log.parent.mkdir(parents=True, exist_ok=True)

    ts = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    clean_reason = " ".join(str(reason or "Variação normal de acabamento").splitlines()).strip()
    if not clean_reason:
        clean_reason = "Variação normal de acabamento"

    row: Dict[str, Any] = {
        "timestamp": ts,
        "category": str(category),
        "previous_score": round(float(previous_score), 4),
        "new_score": round(float(new_score), 4),
        "added_vectors_count": int(added_vectors_count),
        "memory_bank_size_after": int(memory_bank_size_after) if memory_bank_size_after is not None else "",
        "faiss_update_ms": round(float(faiss_update_ms), 2) if faiss_update_ms is not None else "",
        "reason": clean_reason,
        "feedback_image_path": str(feedback_image_path) if feedback_image_path is not None else "",
    }

    write_header = not target_log.exists() or target_log.stat().st_size == 0
    with open(target_log, mode="a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(AUDIT_CSV_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return row


def read_contested_samples_log(
    log_path: Union[str, Path] = DEFAULT_AUDIT_LOG_PATH,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read recorded contestation rows from ``logs/contested_samples.csv``."""
    target_log = Path(log_path)
    if not target_log.is_file():
        return []

    rows: List[Dict[str, Any]] = []
    with open(target_log, mode="r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for item in reader:
            if category is not None and item.get("category") != category:
                continue
            rows.append(dict(item))
    return rows

