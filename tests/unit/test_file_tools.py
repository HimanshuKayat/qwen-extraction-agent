"""Unit tests for tools.file_tools: compact summaries, not full dumps."""

from pathlib import Path

import pandas as pd
import pytest

from core.exceptions import ToolExecutionError
from tools.file_tools import inspect_file, read_csv, read_excel


def test_inspect_file_missing_raises():
    with pytest.raises(ToolExecutionError):
        inspect_file("/tmp/definitely_does_not_exist_12345.csv")


def test_inspect_file_reports_basic_metadata(tmp_path: Path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")

    result = inspect_file(str(file_path))
    assert result["success"] is True
    assert result["extension"] == ".csv"
    assert result["size_bytes"] > 0


def test_read_csv_returns_compact_summary(tmp_path: Path):
    file_path = tmp_path / "data.csv"
    df = pd.DataFrame({"mcc": [5411, 5812], "category": ["Grocery", "Restaurant"]})
    df.to_csv(file_path, index=False)

    result = read_csv(str(file_path))
    assert result["success"] is True
    assert result["shape"] == {"rows": 2, "columns": 2}
    assert set(result["columns"]) == {"mcc", "category"}
    assert len(result["sample_rows"]) == 2


def test_read_csv_never_returns_full_dataframe_object(tmp_path: Path):
    file_path = tmp_path / "big.csv"
    df = pd.DataFrame({"x": range(1000)})
    df.to_csv(file_path, index=False)

    result = read_csv(str(file_path), sample_rows=5)
    assert len(result["sample_rows"]) == 5
    assert result["shape"]["rows"] == 1000


def test_read_excel_returns_compact_summary(tmp_path: Path):
    file_path = tmp_path / "data.xlsx"
    df = pd.DataFrame({"col1": [1, 2, 3]})
    df.to_excel(file_path, index=False, sheet_name="Sheet1")

    result = read_excel(str(file_path))
    assert result["success"] is True
    assert result["sheet_name"] == "Sheet1"
    assert result["shape"]["rows"] == 3
