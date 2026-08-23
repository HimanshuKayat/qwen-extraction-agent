"""Controlled tool registry.

This module defines:

    ToolSpec
        Machine-readable description of a tool.

    ToolRegistry
        Controlled collection of registered tools.

    execute_action
        The single execution gateway used by the agent.

The model can only execute tools explicitly registered here.

The registry supports both:

    1. Synchronous tools
       e.g. http_download, read_pdf, read_excel

    2. Asynchronous tools
       e.g. browser_open, browser_inspect, browser_close

Async tools are executed safely from the synchronous agent interface.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
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
    """Machine-readable description of a single tool."""

    name: str
    description: str
    category: str
    function: Optional[Callable[..., Any]]
    argument_schema: Dict[str, Any]
    enabled: bool = True

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Return a compact description suitable for the model prompt."""

        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "arguments": self.argument_schema,
            "enabled": self.enabled,
        }


class ToolRegistry:
    """A controlled collection of tools available to the agent."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool.

        Raises:
            ValueError:
                If a tool with the same name is already registered.
        """

        if spec.name in self._tools:
            raise ValueError(
                f"Tool '{spec.name}' is already registered."
            )

        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """Retrieve a tool by name."""

        if name not in self._tools:
            raise ToolNotFoundError(
                f"Unknown tool: '{name}'"
            )

        return self._tools[name]

    def list_enabled(self) -> List[ToolSpec]:
        """Return only enabled tools."""

        return [
            spec
            for spec in self._tools.values()
            if spec.enabled
        ]

    def list_all(self) -> List[ToolSpec]:
        """Return all registered tools."""

        return list(self._tools.values())

    def to_prompt_list(self) -> List[Dict[str, Any]]:
        """Return enabled tools in model-prompt format."""

        return [
            spec.to_prompt_dict()
            for spec in self.list_enabled()
        ]


def validate_arguments(
    spec: ToolSpec,
    arguments: Dict[str, Any],
) -> None:
    """Validate model-provided arguments against a JSON Schema.

    Raises:
        InvalidArgumentsError:
            If the arguments are not valid for the tool.
    """

    if not isinstance(arguments, dict):
        raise InvalidArgumentsError(
            (
                f"Arguments for tool '{spec.name}' must be "
                f"a JSON object, got {type(arguments).__name__}"
            )
        )

    try:
        jsonschema.validate(
            instance=arguments,
            schema=spec.argument_schema,
        )

    except JsonSchemaValidationError as exc:
        raise InvalidArgumentsError(
            (
                f"Invalid arguments for tool "
                f"'{spec.name}': {exc.message}"
            )
        ) from exc


def _run_async_result(result: Any) -> Any:
    """Execute an awaitable from the synchronous agent interface.

    The agent currently exposes a synchronous execute_action() API,
    while browser tools use Playwright's asynchronous API.

    If no event loop is currently running, asyncio.run() is safe.

    If an event loop is already running, we deliberately refuse to
    execute the coroutine because nesting event loops in the same
    thread can cause subtle browser/session problems.
    """

    try:
        asyncio.get_running_loop()

    except RuntimeError:
        return asyncio.run(result)

    raise ToolExecutionError(
        message=(
            "An asynchronous tool cannot be executed through the "
            "synchronous agent interface while an event loop is "
            "already running."
        ),
        error_type="AsyncExecutionContextError",
        recoverable=False,
    )


def _execute_function(
    function: Callable[..., Any],
    arguments: Dict[str, Any],
) -> Any:
    """Execute either a synchronous or asynchronous tool function."""

    result = function(**arguments)

    if inspect.isawaitable(result):
        return _run_async_result(result)

    return result


def execute_action(
    registry: ToolRegistry,
    action: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate and execute one model-selected action.

    Execution order:

        1. Handle ``finish``.
        2. Look up the requested tool.
        3. Verify that it is enabled.
        4. Verify that it has an implementation.
        5. Validate arguments.
        6. Execute synchronous or asynchronous tool.
        7. Convert tool failures into structured results.

    The model therefore never gets direct access to:

        eval()
        exec()
        shell commands
        arbitrary Python
        arbitrary filesystem execution

    Returns:
        A structured dictionary containing at least ``success``.
    """

    # ---------------------------------------------------------------
    # Special finish action
    # ---------------------------------------------------------------

    if action == FINISH_ACTION:
        reason = (
            arguments.get("reason", "")
            if isinstance(arguments, dict)
            else ""
        )

        return {
            "success": True,
            "action": FINISH_ACTION,
            "reason": reason,
        }

    # ---------------------------------------------------------------
    # Tool lookup
    # ---------------------------------------------------------------

    spec = registry.get(action)

    # ---------------------------------------------------------------
    # Enabled check
    # ---------------------------------------------------------------

    if not spec.enabled:
        raise ToolDisabledError(
            (
                f"Tool '{action}' is registered but "
                "not currently enabled."
            )
        )

    # ---------------------------------------------------------------
    # Implementation check
    # ---------------------------------------------------------------

    if spec.function is None:
        raise ToolDisabledError(
            (
                f"Tool '{action}' has no implementation "
                "and cannot be executed."
            )
        )

    # ---------------------------------------------------------------
    # Argument validation
    # ---------------------------------------------------------------

    validate_arguments(
        spec,
        arguments,
    )

    # ---------------------------------------------------------------
    # Tool execution
    # ---------------------------------------------------------------

    start = time.monotonic()

    try:
        result = _execute_function(
            spec.function,
            arguments,
        )

    except ToolExecutionError as exc:
        duration = time.monotonic() - start

        return {
            "success": False,
            "error_type": exc.error_type,
            "message": exc.message,
            "recoverable": exc.recoverable,
            "duration_seconds": round(
                duration,
                4,
            ),
        }

    except Exception as exc:
        # Tool failures must never crash the agent.
        duration = time.monotonic() - start

        return {
            "success": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "recoverable": True,
            "duration_seconds": round(
                duration,
                4,
            ),
        }

    # ---------------------------------------------------------------
    # Normalize tool result
    # ---------------------------------------------------------------

    duration = time.monotonic() - start

    if isinstance(result, dict):
        result.setdefault(
            "success",
            True,
        )

        result["duration_seconds"] = round(
            duration,
            4,
        )

        return result

    # Defensive wrapping for tools that accidentally return
    # something other than a dictionary.

    return {
        "success": True,
        "result": result,
        "duration_seconds": round(
            duration,
            4,
        ),
    }
