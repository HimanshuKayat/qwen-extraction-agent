"""Dedicated unit test for the 'unknown tool' failure path, per the
Phase 1 test checklist (spec section 23, item 5).
"""

import pytest

from core.exceptions import ToolNotFoundError
from tools.definitions import build_registry
from tools.registry import execute_action


def test_unknown_tool_name_raises_tool_not_found():
    registry = build_registry()
    with pytest.raises(ToolNotFoundError):
        execute_action(registry, "totally_made_up_tool", {"foo": "bar"})


def test_registry_get_unknown_tool_raises_tool_not_found():
    registry = build_registry()
    with pytest.raises(ToolNotFoundError):
        registry.get("totally_made_up_tool")
