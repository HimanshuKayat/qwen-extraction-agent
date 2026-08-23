"""Unit tests for tools.registry.execute_action.

These tests verify that model-selected actions are:

- correctly routed to registered tools
- rejected when unknown
- rejected when disabled
- rejected when arguments are invalid
- converted into structured failures when tools crash
"""

import pytest

from core.exceptions import (
    InvalidArgumentsError,
    ToolDisabledError,
    ToolNotFoundError,
)
from tools.definitions import build_registry
from tools.registry import execute_action


def test_execute_finish_action_does_not_touch_tools():
    """The finish action should succeed without calling a tool."""

    registry = build_registry()

    result = execute_action(
        registry,
        "finish",
        {
            "reason": "Done.",
        },
    )

    assert result["success"] is True
    assert result["action"] == "finish"
    assert result["reason"] == "Done."


def test_execute_unknown_tool_raises():
    """An action that does not exist must raise ToolNotFoundError."""

    registry = build_registry()

    with pytest.raises(ToolNotFoundError):
        execute_action(
            registry,
            "does_not_exist",
            {},
        )


def test_execute_disabled_future_tool_raises():
    """A registered but disabled future tool must raise ToolDisabledError."""

    registry = build_registry()

    # browser_click is registered as a future placeholder.
    # It exists, but is intentionally disabled.
    with pytest.raises(ToolDisabledError):
        execute_action(
            registry,
            "browser_click",
            {},
        )


def test_execute_invalid_arguments_raises():
    """Missing required arguments must be rejected."""

    registry = build_registry()

    with pytest.raises(InvalidArgumentsError):
        execute_action(
            registry,
            "http_download",
            {
                "url": "https://example.com",
            },
        )


def test_execute_invalid_argument_types_raises():
    """Arguments with invalid JSON types must be rejected."""

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


def test_execute_tool_runtime_error_is_structured_not_raised():
    """A tool runtime failure must become a structured result."""

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
    assert "duration_seconds" in result


def test_execute_browser_open_invalid_arguments_raises():
    """browser_open requires a URL."""

    registry = build_registry()

    with pytest.raises(InvalidArgumentsError):
        execute_action(
            registry,
            "browser_open",
            {},
        )


def test_execute_browser_inspect_accepts_empty_arguments():
    """browser_inspect accepts an empty argument object."""

    registry = build_registry()

    # We don't actually execute Playwright here.
    # Replace the implementation with a synchronous test double.
    registry.get(
        "browser_inspect"
    ).function = lambda: {
        "success": True,
        "title": "Test page",
    }

    result = execute_action(
        registry,
        "browser_inspect",
        {},
    )

    assert result["success"] is True
    assert result["title"] == "Test page"


def test_execute_browser_close_accepts_empty_arguments():
    """browser_close accepts an empty argument object."""

    registry = build_registry()

    registry.get(
        "browser_close"
    ).function = lambda: {
        "success": True,
        "message": "Browser session closed.",
    }

    result = execute_action(
        registry,
        "browser_close",
        {},
    )

    assert result["success"] is True
    assert result["message"] == "Browser session closed."
