"""Controlled tool registry.

This module defines:

  * ``ToolSpec``      - a machine-readable description of one tool
                         (name, description, category, argument schema,
                         function, enabled flag).
  * ``ToolRegistry``   - a container of ToolSpecs with lookup helpers.
  * ``execute_action`` - the single, controlled entry point the agent
                         loop uses to run a model-selected action.

The model must NEVER be able to call eval(), exec(), arbitrary shell
commands, or arbitrary Python. Every tool it can reach is explicitly
registered here with an argument schema that is validated before the
underlying function ever runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import jsonschema
from jsonschema import ValidationError as JsonSchemaValidationError

from core.exceptions import (
    InvalidArgumentsError,
    ToolDisabledError,
    ToolExecutionError,
    ToolNotFoundError,
)

FINISH_ACTION = "finish"


@dataclass
class ToolSpec:
    """Machine-readable description of a single tool.

    ``argument_schema`` is a JSON Schema (draft-07 compatible subset)
    describing the accepted arguments. It is validated against the
    model-supplied arguments before ``function`` is ever invoked.
    """

    name: str
    description: str
    category: str
    function: Optional[Callable[..., Dict[str, Any]]]
    argument_schema: Dict[str, Any]
    enabled: bool = True

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Compact, machine-readable description suitable for showing to the model."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "arguments": self.argument_schema,
            "enabled": self.enabled,
        }


class ToolRegistry:
    """A controlled collection of tools the agent may invoke."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered.")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolNotFoundError(f"Unknown tool: '{name}'")
        return self._tools[name]

    def list_enabled(self) -> List[ToolSpec]:
        return [spec for spec in self._tools.values() if spec.enabled]

    def list_all(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def to_prompt_list(self) -> List[Dict[str, Any]]:
        """Machine-readable description of all enabled tools, for the model prompt."""
        return [spec.to_prompt_dict() for spec in self.list_enabled()]


def validate_arguments(spec: ToolSpec, arguments: Dict[str, Any]) -> None:
    """Validate ``arguments`` against ``spec.argument_schema``.

    Raises:
        InvalidArgumentsError: If validation fails.
    """
    if not isinstance(arguments, dict):
        raise InvalidArgumentsError(
            f"Arguments for tool '{spec.name}' must be a JSON object, got {type(arguments).__name__}"
        )
    try:
        jsonschema.validate(instance=arguments, schema=spec.argument_schema)
    except JsonSchemaValidationError as exc:
        raise InvalidArgumentsError(
            f"Invalid arguments for tool '{spec.name}': {exc.message}"
        ) from exc


def execute_action(
    registry: ToolRegistry,
    action: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate and execute one model-selected action.

    This is the ONLY path by which a model decision turns into real
    execution. It performs, in order:

      1. Special-cases the 'finish' action (no function execution).
      2. Looks up the tool in the registry (raises ToolNotFoundError).
      3. Confirms the tool is enabled (raises ToolDisabledError).
      4. Validates arguments against the tool's JSON schema
         (raises InvalidArgumentsError).
      5. Executes the tool's function, catching runtime errors and
         converting them into a structured result rather than letting
         the process crash.

    Returns:
        A structured result dictionary. Always contains at least a
        "success" boolean key.
    """
    if action == FINISH_ACTION:
        reason = arguments.get("reason", "") if isinstance(arguments, dict) else ""
        return {"success": True, "action": FINISH_ACTION, "reason": reason}

    spec = registry.get(action)  # raises ToolNotFoundError

    if not spec.enabled:
        raise ToolDisabledError(f"Tool '{action}' is registered but not yet enabled.")

    if spec.function is None:
        raise ToolDisabledError(
            f"Tool '{action}' is a placeholder for a future phase and has no implementation."
        )

    validate_arguments(spec, arguments)  # raises InvalidArgumentsError

    start = time.monotonic()
    try:
        result = spec.function(**arguments)
    except ToolExecutionError as exc:
        duration = time.monotonic() - start
        return {
            "success": False,
            "error_type": exc.error_type,
            "message": exc.message,
            "recoverable": exc.recoverable,
            "duration_seconds": round(duration, 4),
        }
    except Exception as exc:  # noqa: BLE001 - deliberately broad: tools must never crash the agent
        duration = time.monotonic() - start
        return {
            "success": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "recoverable": True,
            "duration_seconds": round(duration, 4),
        }

    duration = time.monotonic() - start
    if isinstance(result, dict):
        result.setdefault("success", True)
        result["duration_seconds"] = round(duration, 4)
        return result

    # Tools are expected to return dicts; wrap anything else defensively.
    return {"success": True, "result": result, "duration_seconds": round(duration, 4)}
 def build_registry() -> ToolRegistry:
    """
    Build the currently enabled Phase-1 tool registry.

    This is the single entry point used by the agent
    to determine which deterministic capabilities are
    currently available.

    Future tools such as Playwright, email, SPARQL,
    and API tools will be added here only when their
    implementations are ready.
    """

    from tools.definitions import (
        get_phase1_tool_specs,
    )

    registry = ToolRegistry()

    for spec in get_phase1_tool_specs():
        registry.register(spec)

    return registry
