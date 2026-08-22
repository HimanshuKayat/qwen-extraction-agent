"""Unit tests for tools.registry.execute_action."""

import pytest

from core.exceptions import (
    InvalidArgumentsError,
    ToolDisabledError,
    ToolNotFoundError,
)
from tools.definitions import build_registry
from tools.registry import execute_action


def test_execute_finish_action_does_not_touch_tools():
    registry = build_registry()

    result = execute_action(
        registry,
        "finish",
        {"reason": "done"},
    )

    assert result["success"] is True
    assert result["action"] == "finish"
    assert result["reason"] == "done"


def test_execute_unknown_tool_raises():
    registry = build_registry()

    with pytest.raises(ToolNotFoundError):
        execute_action(
            registry,
            "does_not_exist",
            {},
        )


def test_execute_disabled_future_tool_raises():
    """Future tools are not registered until implemented.

    Therefore an unavailable future tool currently raises
    ToolNotFoundError rather than executing anything.
    """
    registry = build_registry()

    with pytest.raises(ToolNotFoundError):
        execute_action(
            registry,
            "browser_open",
            {"url": "https://example.com"},
        )


def test_execute_invalid_arguments_raises():
    registry = build_registry()

    with pytest.raises(InvalidArgumentsError):
        execute_action(
            registry,
            "http_download",
            {
                "url": "https://example.com",
                "save_path": "invalid.bin",
            },
        )


def test_execute_invalid_argument_types_raises():
    registry = build_registry()

    with pytest.raises(InvalidArgumentsError):
        execute_action(
            registry,
            "http_download",
            {
                "url": "https://example.com",
                "source_id": 123,
                "filename": "test.bin",
            },
        )


def test_execute_tool_runtime_error_is_structured_not_raised(
    monkeypatch,
):
    """A tool that raises at runtime should produce a structured
    failure result instead of crashing execute_action.
    """

    registry = build_registry()

    def broken_download(**kwargs):
        raise RuntimeError("simulated failure")

    registry.get(
        "http_download"
    ).function = broken_download

    result = execute_action(
        registry,
        "http_download",
        {
            "url": "https://example.com",
            "source_id": "test_source",
            "filename": "test.bin",
        },
    )

    assert result["success"] is False
    assert result["error_type"] == "RuntimeError"
    assert result["message"] == "simulated failure"
    assert result["recoverable"] is True
