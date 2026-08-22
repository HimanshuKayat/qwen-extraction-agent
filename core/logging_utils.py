"""Structured, auditable logging for tool calls and agent events.

This is intentionally simple: it writes newline-delimited JSON to a log
file plus stdout. A full monitoring platform is explicitly out of scope
for the Phase 1 foundation (see spec section 21).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_LOG_DIR = Path("storage/logs")


def get_logger(name: str = "agent") -> logging.Logger:
    """Return a configured stdlib logger.

    Safe to call repeatedly; handlers are only attached once per name.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def log_tool_call(
    source_id: str,
    step: int,
    tool_name: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any],
    duration_seconds: float,
    error: Optional[str] = None,
    log_dir: Path = DEFAULT_LOG_DIR,
) -> None:
    """Append one structured tool-call audit record as a JSON line.

    Fields recorded: timestamp, source_id, agent step, tool name,
    arguments, result status, error (if any), and duration.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "tool_calls.jsonl"

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_id": source_id,
        "step": step,
        "tool_name": tool_name,
        "arguments": arguments,
        "success": bool(result.get("success", False)) if isinstance(result, dict) else False,
        "error": error,
        "duration_seconds": round(duration_seconds, 4),
    }

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
