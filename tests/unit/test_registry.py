"""Unit tests for the controlled tool registry.

The registry is responsible for:
- registering deterministic tools;
- preventing duplicate registrations;
- looking up tools;
- exposing only enabled tools to the model;
- validating the structure of tool definitions.

Only currently implemented tools should be present in the Phase-1
registry. Future tools will be added when their implementations exist.
"""

import pytest

from core.exceptions import ToolNotFoundError
from tools.definitions import build_registry
from tools.registry import ToolRegistry, ToolSpec


def test_build_registry_contains_phase1_tools():
    """The Phase-1 registry contains all currently implemented tools."""

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

    assert names == expected_tools


def test_phase1_tools_are_enabled():
    """Every Phase-1 tool is enabled and has an implementation."""

    registry = build_registry()

    for tool_name in (
        "http_download",
        "inspect_file",
        "read_csv",
        "read_excel",
        "read_pdf",
        "extract_pdf_table",
        "validate_required_fields",
        "validate_row_count",
    ):
        spec = registry.get(tool_name)

        assert spec.enabled is True
        assert spec.function is not None


def test_list_enabled_contains_all_phase1_tools():
    """All registered Phase-1 tools are visible to the agent."""

    registry = build_registry()

    enabled_names = {
        spec.name
        for spec in registry.list_enabled()
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

    assert enabled_names == expected_tools


def test_get_unknown_tool_raises():
    """Unknown tool names must never resolve successfully."""

    registry = build_registry()

    with pytest.raises(ToolNotFoundError):
        registry.get("does_not_exist")


def test_duplicate_registration_raises():
    """Registering the same tool twice must fail."""

    registry = ToolRegistry()

    spec = ToolSpec(
        name="dummy",
        description="dummy tool",
        category="test",
        function=lambda: {"success": True},
        argument_schema={
            "type": "object",
            "properties": {},
        },
    )

    registry.register(spec)

    with pytest.raises(ValueError):
        registry.register(spec)


def test_to_prompt_list_is_json_serializable_shape():
    """Tool descriptions exposed to the model must be JSON serializable."""

    import json

    registry = build_registry()

    prompt_list = registry.to_prompt_list()

    assert isinstance(prompt_list, list)

    assert len(prompt_list) == 8

    for entry in prompt_list:
        assert isinstance(entry, dict)

        assert "name" in entry
        assert "description" in entry
        assert "category" in entry
        assert "arguments" in entry
        assert "enabled" in entry

        assert isinstance(entry["name"], str)
        assert isinstance(entry["description"], str)
        assert isinstance(entry["category"], str)
        assert isinstance(entry["arguments"], dict)
        assert entry["enabled"] is True

    # The complete tool description must be safely serializable
    # before it is passed into a model prompt.
    json.dumps(
        prompt_list,
        ensure_ascii=False,
    )


def test_tool_names_are_unique():
    """The registry must never contain duplicate tool names."""

    registry = build_registry()

    names = [
        spec.name
        for spec in registry.list_all()
    ]

    assert len(names) == len(set(names))


def test_all_tools_have_argument_schemas():
    """Every registered tool must define a JSON Schema."""

    registry = build_registry()

    for spec in registry.list_all():
        assert isinstance(
            spec.argument_schema,
            dict,
        )

        assert spec.argument_schema.get(
            "type"
        ) == "object"
