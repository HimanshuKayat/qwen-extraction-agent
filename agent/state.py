"""Agent state abstraction.

Tracks everything about a single extraction run: the source configuration,
how many steps have been taken, the history of tool calls and their
results, any artifacts produced, and the final status.

This module contains no model logic and no tool logic. It is a plain,
serializable record of what happened during one agent run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(str, Enum):
    """Lifecycle status of an agent run."""

    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    MAX_STEPS_REACHED = "max_steps_reached"


@dataclass
class ToolCallRecord:
    """A single tool invocation and its outcome."""

    step: int
    action: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    duration_seconds: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "arguments": self.arguments,
            "result": self.result,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentState:
    """Mutable state for one agent run against one source configuration."""

    source_config: Dict[str, Any]
    max_steps: int = 20
    current_step: int = 0
    status: AgentStatus = AgentStatus.RUNNING
    tool_history: List[ToolCallRecord] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None

    def record_tool_call(
        self,
        action: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        duration_seconds: float,
    ) -> None:
        """Append a completed tool call to the history."""
        record = ToolCallRecord(
            step=self.current_step,
            action=action,
            arguments=arguments,
            result=result,
            duration_seconds=duration_seconds,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        self.tool_history.append(record)

        if isinstance(result, dict) and result.get("success"):
            file_path = result.get("file_path")
            if file_path:
                self.artifacts.append(file_path)
        elif isinstance(result, dict) and not result.get("success", True):
            message = result.get("message", "unknown error")
            self.errors.append(f"step={self.current_step} action={action}: {message}")

    def add_observation(self, text: str) -> None:
        self.observations.append(text)

    def advance_step(self) -> None:
        self.current_step += 1
        if self.current_step >= self.max_steps and self.status == AgentStatus.RUNNING:
            self.status = AgentStatus.MAX_STEPS_REACHED

    def finish(self, reason: str) -> None:
        self.status = AgentStatus.FINISHED
        self.finish_reason = reason

    def fail(self, reason: str) -> None:
        self.status = AgentStatus.FAILED
        self.finish_reason = reason
        self.errors.append(reason)

    def is_active(self) -> bool:
        return self.status == AgentStatus.RUNNING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_config": self.source_config,
            "max_steps": self.max_steps,
            "current_step": self.current_step,
            "status": self.status.value,
            "tool_history": [record.to_dict() for record in self.tool_history],
            "observations": self.observations,
            "artifacts": self.artifacts,
            "errors": self.errors,
            "validation_results": self.validation_results,
            "finish_reason": self.finish_reason,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
