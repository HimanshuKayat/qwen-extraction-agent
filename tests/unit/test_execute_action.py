"""Unit tests for tools.registry.execute_action: the single controlled
entry point from model decision to real execution.
"""

import pytest

from core.exceptions import InvalidArgumentsError, ToolDisabledError, ToolNotFoundError
from tools.definitions import build_registry
from tools.registry import FINISH_ACTION, execute_action


def test_execute_finish_action_does_not_touch_tools():
    registry = build_registry()
    result = execute_action(registry, FINISH_ACTION, {"reason": "done"})
    assert result["success"] is True
    assert result["action"] == FINISH_ACTION
    assert result["reason"] == "done"


def test_execute_unknown_tool_raises():
    registry = build_registry()
    with pytest.raises(ToolNotFoundError):
        execute_action(registry, "not_a_real_tool", {})


def test_execute_disabled_future_tool_raises():
    registry = build_registry()
    with pytest.raises(ToolDisabledError):
        execute_action(registry, "browser_open", {"url": "https://example.com"})


def test_execute_invalid_arguments_raises():
    registry = build_registry()
    # http_download requires 'url' and 'save_path'; omit both.
    with pytest.raises(InvalidArgumentsError):
        execute_action(registry, "http_download", {})


def test_execute_invalid_argument_types_raises():
    registry = build_registry()
    with pytest.raises(InvalidArgumentsError):
        execute_action(registry, "http_download", {"url": 123, "save_path": "x"})


def test_execute_tool_runtime_error_is_structured_not_raised(monkeypatch):
    """A tool that raises at runtime should produce a structured failure
    result, not crash execute_action.
    """
    registry = build_registry()

    def broken_download(**kwargs):
        raise RuntimeError("simulated failure")

    registry.get("http_download").function = broken_download

    result = execute_action(registry, "http_download", {"url": "https://example.com", "save_path": "x/y.bin"})
    assert result["success"] is False
    assert result["error_type"] == "RuntimeError"
    assert "simulated failure" in result["message"]
