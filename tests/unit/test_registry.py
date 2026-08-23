"""Unit tests for the tool registry.

Tests cover:

- tool registration
- tool lookup
- duplicate protection
- enabled/disabled tools
- Phase 1 extraction tools
- Phase 2 browser tools
- future placeholder tools
- model-facing tool descriptions
"""

import json

import pytest

from core.exceptions import ToolNotFoundError
from tools.definitions import build_registry
from tools.registry import ToolRegistry, ToolSpec


def test_build_registry_contains_phase1_tools():
    """The registry contains all implemented Phase-1 tools."""

    registry = build_registry()

    names = {
        spec.name
        for spec in registry.list_all()
    }

    expected_tools = {
        "http_download",
        "inspect_file",
        "read_csv",
        "read_excel",
        "read_pdf",
        "extract_pdf_table",
        "validate_required_fields",
        "validate_row_count",
    }

    assert expected_tools.issubset(names)


def test_build_registry_contains_phase2_browser_tools():
    """The registry contains the implemented browser tools."""

    registry = build_registry()

    names = {
        spec.name
        for spec in registry.list_all()
    }

    expected_browser_tools = {
        "browser_open",
        "browser_inspect",
        "browser_close",
    }

    assert expected_browser_tools.issubset(names)


def test_phase1_tools_are_enabled():
    """All Phase-1 tools are enabled."""

    registry = build_registry()

    phase1_tools = [
        "http_download",
        "inspect_file",
        "read_csv",
        "read_excel",
        "read_pdf",
        "extract_pdf_table",
        "validate_required_fields",
        "validate_row_count",
    ]

    for name in phase1_tools:
        spec = registry.get(name)

        assert spec.enabled is True
        assert spec.function is not None


def test_phase2_browser_tools_are_enabled():
    """Implemented browser tools are enabled."""

    registry = build_registry()

    browser_tools = [
        "browser_open",
        "browser_inspect",
        "browser_close",
    ]

    for name in browser_tools:
        spec = registry.get(name)

        assert spec.enabled is True
        assert spec.function is not None


def test_future_tools_are_disabled_placeholders():
    """Future tools remain registered but disabled."""

    registry = build_registry()

    future_tools = [
        "browser_click",
        "browser_fill",
        "browser_select",
        "browser_wait",
        "browser_download",
        "browser_back",
        "sparql_query",
        "api_get",
        "email_search",
        "email_read",
        "email_get_attachment",
        "email_download_link",
    ]

    for name in future_tools:
        spec = registry.get(name)

        assert spec.enabled is False
        assert spec.function is None


def test_get_unknown_tool_raises():
    """Unknown tools raise ToolNotFoundError."""

    registry = build_registry()

    with pytest.raises(ToolNotFoundError):
        registry.get("does_not_exist")


def test_list_enabled_contains_all_implemented_tools():
    """All implemented tools are visible to the agent."""

    registry = build_registry()

    enabled_names = {
        spec.name
        for spec in registry.list_enabled()
    }

    expected_enabled_tools = {
        # Phase 1
        "http_download",
        "inspect_file",
        "read_csv",
        "read_excel",
        "read_pdf",
        "extract_pdf_table",
        "validate_required_fields",
        "validate_row_count",

        # Phase 2
        "browser_open",
        "browser_inspect",
        "browser_close",
    }

    assert enabled_names == expected_enabled_tools


def test_list_enabled_excludes_future_placeholders():
    """Disabled future tools must not be exposed to the model."""

    registry = build_registry()

    enabled_names = {
        spec.name
        for spec in registry.list_enabled()
    }

    future_tools = {
        "browser_click",
        "browser_fill",
        "browser_select",
        "browser_wait",
        "browser_download",
        "browser_back",
        "sparql_query",
        "api_get",
        "email_search",
        "email_read",
        "email_get_attachment",
        "email_download_link",
    }

    assert enabled_names.isdisjoint(future_tools)


def test_duplicate_registration_raises():
    """Registering the same tool twice raises ValueError."""

    registry = ToolRegistry()

    spec = ToolSpec(
        name="dummy",
        description="dummy tool",
        category="test",
        function=lambda: {
            "success": True,
        },
        argument_schema={
            "type": "object",
            "properties": {},
        },
    )

    registry.register(spec)

    with pytest.raises(ValueError):
        registry.register(spec)


def test_to_prompt_list_is_json_serializable_shape():
    """Enabled tool descriptions must be JSON serializable."""

    registry = build_registry()

    prompt_list = registry.to_prompt_list()

    assert isinstance(prompt_list, list)

    # 8 Phase-1 tools + 3 implemented Phase-2 browser tools.
    assert len(prompt_list) == 11

    for entry in prompt_list:
        assert "name" in entry
        assert "description" in entry
        assert "category" in entry
        assert "arguments" in entry
        assert "enabled" in entry

        assert entry["enabled"] is True

    # Ensure the complete prompt structure is JSON serializable.
    json.dumps(prompt_list)


def test_tool_names_are_unique():
    """Every registered tool must have a unique name."""

    registry = build_registry()

    names = [
        spec.name
        for spec in registry.list_all()
    ]

    assert len(names) == len(set(names))


def test_all_tools_have_argument_schemas():
    """Every registered tool must expose an argument schema."""

    registry = build_registry()

    for spec in registry.list_all():
        assert isinstance(
            spec.argument_schema,
            dict,
        )

        assert spec.argument_schema.get("type") == "object"


def test_browser_open_has_url_argument():
    """browser_open must expose the URL argument to the model."""

    registry = build_registry()

    spec = registry.get("browser_open")

    properties = spec.argument_schema["properties"]

    assert "url" in properties
    assert "timeout" in properties

    assert "url" in spec.argument_schema["required"]


def test_browser_inspect_requires_no_arguments():
    """browser_inspect should accept an empty argument object."""

    registry = build_registry()

    spec = registry.get("browser_inspect")

    assert spec.argument_schema["type"] == "object"
    assert spec.argument_schema["properties"] == {}
    assert spec.argument_schema["additionalProperties"] is False


def test_browser_close_requires_no_arguments():
    """browser_close should accept an empty argument object."""

    registry = build_registry()

    spec = registry.get("browser_close")

    assert spec.argument_schema["type"] == "object"
    assert spec.argument_schema["properties"] == {}
    assert spec.argument_schema["additionalProperties"] is False
