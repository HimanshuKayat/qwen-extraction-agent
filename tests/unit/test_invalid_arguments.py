"""Dedicated unit test for the 'invalid arguments' failure path, per the
Phase 1 test checklist (spec section 23, item 6).
"""

import pytest

from core.exceptions import InvalidArgumentsError
from tools.definitions import build_registry
from tools.registry import execute_action, validate_arguments


def test_missing_required_argument_raises():
    registry = build_registry()
    with pytest.raises(InvalidArgumentsError):
        execute_action(registry, "http_download", {"url": "https://example.com"})  # missing save_path


def test_unexpected_extra_argument_raises():
    registry = build_registry()
    with pytest.raises(InvalidArgumentsError):
        execute_action(
            registry,
            "http_download",
            {"url": "https://example.com", "save_path": "x.bin", "not_a_real_arg": True},
        )


def test_arguments_not_a_dict_raises():
    registry = build_registry()
    spec = registry.get("http_download")
    with pytest.raises(InvalidArgumentsError):
        validate_arguments(spec, ["not", "a", "dict"])  # type: ignore[arg-type]
