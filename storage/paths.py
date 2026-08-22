"""Path helpers enforcing the raw-vs-processed storage principle.

Raw artifacts (as downloaded) must never be overwritten by processed
output. This module centralizes the two roots so tool and pipeline code
never hard-codes "storage/raw" or "storage/processed" as string literals.
"""

from __future__ import annotations

from pathlib import Path

STORAGE_ROOT = Path("storage")
RAW_ROOT = STORAGE_ROOT / "raw"
PROCESSED_ROOT = STORAGE_ROOT / "processed"
LOGS_ROOT = STORAGE_ROOT / "logs"


def raw_path(source_id: str, filename: str) -> Path:
    """Return the raw-artifact path for a given source and filename."""
    return RAW_ROOT / source_id / filename


def processed_path(source_id: str, filename: str) -> Path:
    """Return the processed-data path for a given source and filename."""
    return PROCESSED_ROOT / source_id / filename


def ensure_storage_dirs() -> None:
    """Create the standard storage directories if they do not exist."""
    for root in (RAW_ROOT, PROCESSED_ROOT, LOGS_ROOT):
        root.mkdir(parents=True, exist_ok=True)
