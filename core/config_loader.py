"""Loads and validates source configuration YAML files.

Source configurations describe a data source declaratively. They never
contain executable Python. See config/sources/test_direct_download.yaml
for an example.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from core.exceptions import SourceConfigError

REQUIRED_FIELDS = ("source_id", "source_type", "title", "data_link", "raw_dir")


def load_source_config(path: str | Path) -> Dict[str, Any]:
    """Load a single source configuration file and validate required fields.

    Args:
        path: Path to a YAML source configuration file.

    Returns:
        The parsed configuration as a dictionary. ``target_schema``
        defaults to an empty dict if not present.

    Raises:
        SourceConfigError: If the file is missing, not valid YAML, not a
            mapping, or missing a required field.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise SourceConfigError(f"Source configuration not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise SourceConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SourceConfigError(f"Source configuration must be a mapping: {config_path}")

    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise SourceConfigError(
            f"Source configuration {config_path} is missing required fields: {missing}"
        )

    raw.setdefault("target_schema", {})
    return raw
