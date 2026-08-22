"""Unit tests for the tool registry: registration, lookup, and the
enabled/disabled distinction between Phase 1 and future-phase tools.
"""

import pytest

from core.exceptions import ToolNotFoundError
from tools.definitions import build_registry
from tools.registry import ToolRegistry, ToolSpec


def test_build_registry_contains_phase1_tools():
    registry = build_registry()
    names = {spec.name for spec in registry.list_all()}

    for expected in ("http_download", "inspect_file", "read_csv", "read_excel", "read_pdf", "extract_pdf_table"):
        assert expected in names


def test_phase1_tools_are_enabled():
    registry = build_registry()
    assert registry.get("http_download").enabled is True
    assert registry.get("http_download").function is not None


def test_future_tools_are_disabled_placeholders():
    registry = build_registry()
    browser_open = registry.get("browser_open")
    assert browser_open.enabled is False
    assert browser_open.function is None


def test_get_unknown_tool_raises():
    registry = build_registry()
    with pytest.raises(ToolNotFoundError):
        registry.get("does_not_exist")


def test_list_enabled_excludes_placeholders():
    registry = build_registry()
    enabled_names = {spec.name for spec in registry.list_enabled()}
    assert "http_download" in enabled_names
    assert "browser_open" not in enabled_names


def test_duplicate_registration_raises():
    registry = ToolRegistry()
    spec = ToolSpec(
        name="dummy",
        description="dummy tool",
        category="test",
        function=lambda: {"success": True},
        argument_schema={"type": "object", "properties": {}},
    )
    registry.register(spec)
    with pytest.raises(ValueError):
        registry.register(spec)


def test_to_prompt_list_is_json_serializable_shape():
    registry = build_registry()
    prompt_list = registry.to_prompt_list()
    assert isinstance(prompt_list, list)
    assert all("name" in entry and "arguments" in entry for entry in prompt_list)
