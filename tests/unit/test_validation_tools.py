"""Unit tests for tools.validation_tools."""

from tools.validation_tools import (
    validate_duplicates,
    validate_nulls,
    validate_required_fields,
    validate_row_count,
    validate_schema,
)


def test_validate_required_fields_passes():
    result = validate_required_fields(["a", "b", "c"], ["a", "b"])
    assert result["passed"] is True
    assert result["missing_fields"] == []


def test_validate_required_fields_fails():
    result = validate_required_fields(["a"], ["a", "b"])
    assert result["passed"] is False
    assert result["missing_fields"] == ["b"]


def test_validate_schema_detects_missing_and_extra():
    result = validate_schema(["a", "extra"], {"a": "string", "b": "int"})
    assert result["passed"] is False
    assert result["missing_fields"] == ["b"]
    assert "extra" in result["extra_fields"]


def test_validate_row_count_within_range():
    result = validate_row_count(50, minimum=1, maximum=100)
    assert result["passed"] is True


def test_validate_row_count_below_minimum():
    result = validate_row_count(0, minimum=1)
    assert result["passed"] is False


def test_validate_duplicates_detects_repeats():
    rows = [{"id": 1}, {"id": 2}, {"id": 1}]
    result = validate_duplicates(rows, key_field="id")
    assert result["passed"] is False
    assert 1 in result["duplicate_values_in_sample"]


def test_validate_nulls_detects_missing_values():
    rows = [{"name": "a"}, {"name": ""}, {"name": None}]
    result = validate_nulls(rows, required_non_null_fields=["name"])
    assert result["passed"] is False
    assert "name" in result["fields_with_nulls"]
