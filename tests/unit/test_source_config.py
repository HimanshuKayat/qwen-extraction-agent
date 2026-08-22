"""Unit tests for core.config_loader.load_source_config."""

from pathlib import Path

import pytest
import yaml

from core.config_loader import load_source_config
from core.exceptions import SourceConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_real_test_source_config():
    path = REPO_ROOT / "config" / "sources" / "test_direct_download.yaml"
    config = load_source_config(path)

    assert config["source_id"] == "test_direct_download"
    assert config["source_type"] == "1a"
    assert config["data_link"] == "https://httpbin.org/bytes/100"
    assert config["target_schema"] == {}


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(SourceConfigError):
        load_source_config(tmp_path / "does_not_exist.yaml")


def test_load_invalid_yaml_raises(tmp_path: Path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("key: [unclosed", encoding="utf-8")
    with pytest.raises(SourceConfigError):
        load_source_config(bad_file)


def test_load_non_mapping_yaml_raises(tmp_path: Path):
    list_file = tmp_path / "list.yaml"
    list_file.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(SourceConfigError):
        load_source_config(list_file)


def test_load_missing_required_field_raises(tmp_path: Path):
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text(
        yaml.safe_dump({"source_id": "x", "source_type": "1a"}),
        encoding="utf-8",
    )
    with pytest.raises(SourceConfigError):
        load_source_config(incomplete)


def test_target_schema_defaults_to_empty_dict(tmp_path: Path):
    minimal = tmp_path / "minimal.yaml"
    minimal.write_text(
        yaml.safe_dump({
            "source_id": "x",
            "source_type": "1a",
            "title": "X",
            "data_link": "https://example.com",
            "raw_dir": "raw/x",
        }),
        encoding="utf-8",
    )
    config = load_source_config(minimal)
    assert config["target_schema"] == {}
