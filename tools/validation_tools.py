"""Deterministic validation foundations.

These operate on the compact summaries produced by tools/file_tools.py
(columns, dtypes, sample_rows) plus an optional target_schema pulled
from a source configuration. They are intentionally minimal; a fuller
validation framework is a later milestone (see spec section 18).
"""

from __future__ import annotations

from typing import Any, Dict, List


def validate_required_fields(columns: List[str], required_fields: List[str]) -> Dict[str, Any]:
    """Check that every field in ``required_fields`` is present in ``columns``."""
    missing = [field for field in required_fields if field not in columns]
    return {
        "success": True,
        "check": "required_fields",
        "passed": len(missing) == 0,
        "missing_fields": missing,
    }


def validate_schema(columns: List[str], target_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Check that all columns named in ``target_schema`` are present.

    ``target_schema`` is expected to be a mapping of field name to a
    description of its expected type, e.g. {"mcc": "string"}.
    """
    expected_fields = list(target_schema.keys())
    missing = [field for field in expected_fields if field not in columns]
    extra = [col for col in columns if col not in expected_fields] if expected_fields else []
    return {
        "success": True,
        "check": "schema",
        "passed": len(missing) == 0,
        "missing_fields": missing,
        "extra_fields": extra,
    }


def validate_types(dtypes: Dict[str, str], target_schema: Dict[str, str]) -> Dict[str, Any]:
    """Compare observed pandas dtypes against expected type names in target_schema."""
    mismatches = []
    for field, expected_type in target_schema.items():
        observed_type = dtypes.get(field)
        if observed_type is None:
            continue
        if expected_type.lower() not in observed_type.lower():
            mismatches.append({
                "field": field,
                "expected": expected_type,
                "observed": observed_type,
            })
    return {
        "success": True,
        "check": "types",
        "passed": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def validate_row_count(row_count: int, minimum: int = 1, maximum: int | None = None) -> Dict[str, Any]:
    """Check that a row count falls within an expected range."""
    passed = row_count >= minimum and (maximum is None or row_count <= maximum)
    return {
        "success": True,
        "check": "row_count",
        "passed": passed,
        "row_count": row_count,
        "minimum": minimum,
        "maximum": maximum,
    }


def validate_duplicates(sample_rows: List[Dict[str, Any]], key_field: str) -> Dict[str, Any]:
    """Check for duplicate values of ``key_field`` within a sample of rows.

    This checks only the provided sample, not the full dataset, in
    keeping with the "never dump the full dataset into the model"
    principle; a full-dataset check belongs in the storage/validation
    phase, not this lightweight tool.
    """
    values = [row.get(key_field) for row in sample_rows if key_field in row]
    duplicates = {v for v in values if values.count(v) > 1}
    return {
        "success": True,
        "check": "duplicates",
        "passed": len(duplicates) == 0,
        "duplicate_values_in_sample": list(duplicates),
    }


def validate_date_range(
    sample_rows: List[Dict[str, Any]],
    date_field: str,
    min_date: str | None = None,
    max_date: str | None = None,
) -> Dict[str, Any]:
    """Check that date-like values in a sample fall within an expected range.

    Dates are compared as ISO-format strings (YYYY-MM-DD); this is a
    lightweight check intended for quick sanity checks on a sample, not
    a full date-parsing/validation engine.
    """
    out_of_range = []
    for row in sample_rows:
        value = row.get(date_field)
        if value is None:
            continue
        value_str = str(value)
        if min_date and value_str < min_date:
            out_of_range.append(value_str)
        elif max_date and value_str > max_date:
            out_of_range.append(value_str)
    return {
        "success": True,
        "check": "date_range",
        "passed": len(out_of_range) == 0,
        "out_of_range_values": out_of_range,
    }


def validate_nulls(sample_rows: List[Dict[str, Any]], required_non_null_fields: List[str]) -> Dict[str, Any]:
    """Check that required fields are non-null across a sample of rows."""
    null_fields = set()
    for row in sample_rows:
        for field in required_non_null_fields:
            value = row.get(field)
            if value is None or value == "":
                null_fields.add(field)
    return {
        "success": True,
        "check": "nulls",
        "passed": len(null_fields) == 0,
        "fields_with_nulls": sorted(null_fields),
    }
