from __future__ import annotations

import json
from typing import Any


class AgentResponseError(ValueError):
    """Raised when the model response cannot be parsed."""


def parse_action(response: str) -> dict[str, Any]:
    """
    Parse and validate a model-generated tool action.

    The model is expected to return ONLY JSON.
    """

    if not response or not response.strip():
        raise AgentResponseError(
            "Model returned an empty response."
        )

    cleaned = response.strip()

    # The model is explicitly instructed not to return
    # thinking or markdown. Treat their presence as an
    # invalid response instead of silently modifying it.
    if "<think>" in cleaned or "</think>" in cleaned:
        raise AgentResponseError(
            "Model returned thinking output. "
            "Tool-selection responses must not contain <think>."
        )

    if cleaned.startswith("```"):
        raise AgentResponseError(
            "Model returned markdown instead of raw JSON."
        )

    try:
        action = json.loads(cleaned)

    except json.JSONDecodeError as exc:
        raise AgentResponseError(
            f"Model did not return valid JSON: {exc}"
        ) from exc

    if not isinstance(action, dict):
        raise AgentResponseError(
            "Action must be a JSON object."
        )

    if "action" not in action:
        raise AgentResponseError(
            "Action object is missing 'action'."
        )

    if "arguments" not in action:
        raise AgentResponseError(
            "Action object is missing 'arguments'."
        )

    if not isinstance(action["action"], str):
        raise AgentResponseError(
            "'action' must be a string."
        )

    if not isinstance(action["arguments"], dict):
        raise AgentResponseError(
            "'arguments' must be an object."
        )

    return action