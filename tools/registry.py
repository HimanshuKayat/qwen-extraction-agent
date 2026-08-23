"""Controlled tool registry.

This module defines:

    ToolSpec
        Machine-readable description of a tool.

    ToolRegistry
        Controlled collection of registered tools.

    execute_action
        Synchronous execution gateway.

    execute_action_async
        Asynchronous execution gateway.

The registry supports both:

    1. Synchronous tools
       e.g. http_download, read_pdf, read_excel

    2. Asynchronous tools
       e.g. browser_open, browser_inspect, browser_close

The synchronous gateway is used by the existing synchronous agent.

The asynchronous gateway is used when the agent is running inside an
already-active asyncio environment such as Jupyter/Colab, or when the
agent itself is asynchronous.
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


# ============================================================================
# TOOL SPEC
# ============================================================================


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


# ============================================================================
# TOOL REGISTRY
# ============================================================================


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


# ============================================================================
# ARGUMENT VALIDATION
# ============================================================================


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


# ============================================================================
# FINISH ACTION
# ============================================================================


def _finish_result(
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the standard finish result."""

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


# ============================================================================
# TOOL LOOKUP / VALIDATION
# ============================================================================


def _prepare_tool(
    registry: ToolRegistry,
    action: str,
    arguments: Dict[str, Any],
) -> ToolSpec:
    """Look up and validate a tool before execution."""

    spec = registry.get(action)

    if not spec.enabled:
        raise ToolDisabledError(
            (
                f"Tool '{action}' is registered but "
                "not currently enabled."
            )
        )

    if spec.function is None:
        raise ToolDisabledError(
            (
                f"Tool '{action}' has no implementation "
                "and cannot be executed."
            )
        )

    validate_arguments(
        spec,
        arguments,
    )

    return spec


# ============================================================================
# SYNCHRONOUS TOOL EXECUTION
# ============================================================================


def _execute_sync_function(
    function: Callable[..., Any],
    arguments: Dict[str, Any],
) -> Any:
    """Execute a tool from the synchronous gateway.

    Synchronous tools execute normally.

    Asynchronous tools are allowed only when no event loop is already
    running in the current thread. In Jupyter/Colab, callers should use
    execute_action_async() instead.
    """

    result = function(**arguments)

    if not inspect.isawaitable(result):
        return result

    # We have an async tool.
    try:
        asyncio.get_running_loop()

    except RuntimeError:
        # No running event loop. Safe to use asyncio.run().
        return asyncio.run(result)

    # An event loop is already running. We cannot safely nest it.
    #
    # IMPORTANT:
    # Close the coroutine to prevent:
    #
    # RuntimeWarning:
    # coroutine '...' was never awaited
    #
    try:
        result.close()
    except Exception:
        pass

    raise ToolExecutionError(
        message=(
            "An asynchronous tool cannot be executed through the "
            "synchronous agent interface while an event loop is "
            "already running. Use execute_action_async() instead."
        ),
        error_type="AsyncExecutionContextError",
        recoverable=False,
    )


# ============================================================================
# ASYNC TOOL EXECUTION
# ============================================================================


async def _execute_async_function(
    function: Callable[..., Any],
    arguments: Dict[str, Any],
) -> Any:
    """Execute either a synchronous or asynchronous tool.

    Synchronous functions execute directly.

    Asynchronous functions are awaited directly in the current event loop.
    This is the correct execution path for Playwright tools.
    """

    result = function(**arguments)

    if inspect.isawaitable(result):
        return await result

    return result


# ============================================================================
# RESULT NORMALIZATION
# ============================================================================


def _normalize_result(
    result: Any,
    duration: float,
) -> Dict[str, Any]:
    """Normalize a tool result into a structured dictionary."""

    duration_seconds = round(
        duration,
        4,
    )

    if isinstance(result, dict):
        result.setdefault(
            "success",
            True,
        )

        result["duration_seconds"] = duration_seconds

        return result

    return {
        "success": True,
        "result": result,
        "duration_seconds": duration_seconds,
    }


def _tool_error_result(
    exc: ToolExecutionError,
    duration: float,
) -> Dict[str, Any]:
    """Convert ToolExecutionError into a structured result."""

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


def _unexpected_error_result(
    exc: Exception,
    duration: float,
) -> Dict[str, Any]:
    """Convert unexpected exceptions into structured results."""

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


# ============================================================================
# SYNCHRONOUS EXECUTION GATEWAY
# ============================================================================


def execute_action(
    registry: ToolRegistry,
    action: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate and execute one model-selected action synchronously.

    This is the existing execution gateway used by the synchronous agent.

    Use this for:
        - http_download
        - inspect_file
        - read_csv
        - read_excel
        - read_pdf
        - validation tools
        - other synchronous tools

    For asynchronous browser tools inside Jupyter/Colab or an async agent,
    use execute_action_async().

    Returns:
        A structured dictionary containing at least ``success``.
    """

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    if action == FINISH_ACTION:
        return _finish_result(arguments)

    # ------------------------------------------------------------------
    # Prepare tool
    # ------------------------------------------------------------------

    spec = _prepare_tool(
        registry,
        action,
        arguments,
    )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    start = time.monotonic()

    try:
        result = _execute_sync_function(
            spec.function,
            arguments,
        )

    except ToolExecutionError as exc:
        duration = time.monotonic() - start

        return _tool_error_result(
            exc,
            duration,
        )

    except Exception as exc:
        duration = time.monotonic() - start

        return _unexpected_error_result(
            exc,
            duration,
        )

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    duration = time.monotonic() - start

    return _normalize_result(
        result,
        duration,
    )


# ============================================================================
# ASYNCHRONOUS EXECUTION GATEWAY
# ============================================================================


async def execute_action_async(
    registry: ToolRegistry,
    action: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate and execute one model-selected action asynchronously.

    This is the correct gateway for Playwright browser tools.

    It works inside:
        - Jupyter
        - Google Colab
        - asyncio applications
        - asynchronous agent loops

    It can execute both synchronous and asynchronous tools.

    Browser session lifecycle is preserved because all browser calls
    execute inside the same active asyncio environment.

    Returns:
        A structured dictionary containing at least ``success``.
    """

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    if action == FINISH_ACTION:
        return _finish_result(arguments)

    # ------------------------------------------------------------------
    # Prepare tool
    # ------------------------------------------------------------------

    spec = _prepare_tool(
        registry,
        action,
        arguments,
    )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    start = time.monotonic()

    try:
        result = await _execute_async_function(
            spec.function,
            arguments,
        )

    except ToolExecutionError as exc:
        duration = time.monotonic() - start

        return _tool_error_result(
            exc,
            duration,
        )

    except Exception as exc:
        duration = time.monotonic() - start

        return _unexpected_error_result(
            exc,
            duration,
        )

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    duration = time.monotonic() - start

    return _normalize_result(
        result,
        duration,
    )
