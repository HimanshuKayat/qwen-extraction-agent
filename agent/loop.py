"""The agent control loop.

SOURCE CONFIG -> model -> JSON action -> execute_action -> tool result
-> model -> next JSON action -> ... -> finish

This is a SINGLE agent. There are no sub-agents, planners, or
specialized roles. The model is the only decision maker; every other
component here is deterministic plumbing.
"""

from __future__ import annotations

from typing import Any, Dict

from agent.model import ModelClient
from agent.prompts import build_system_prompt, build_user_prompt, parse_model_action
from agent.state import AgentState
from core.exceptions import (
    InvalidArgumentsError,
    ModelOutputParseError,
    ToolDisabledError,
    ToolNotFoundError,
)
from core.logging_utils import get_logger, log_tool_call
from tools.registry import FINISH_ACTION, ToolRegistry, execute_action

logger = get_logger("agent.loop")


def run_agent(
    source_config: Dict[str, Any],
    registry: ToolRegistry,
    model_client: ModelClient,
    max_steps: int = 20,
) -> AgentState:
    """Run the agent loop for one source configuration until it finishes,
    fails, or reaches ``max_steps``.

    Args:
        source_config: A loaded source configuration (see core.config_loader).
        registry: The controlled tool registry.
        model_client: Anything satisfying agent.model.ModelClient, i.e.
            has a ``.generate(system_prompt, user_prompt, mode)`` method.
        max_steps: Safety cap on the number of tool-selection turns.

    Returns:
        The final AgentState, including full tool history and status.
    """
    state = AgentState(source_config=source_config, max_steps=max_steps)
    system_prompt = build_system_prompt(registry.to_prompt_list())
    source_id = source_config.get("source_id", "unknown_source")

    while state.is_active():
        user_prompt = build_user_prompt(
            source_config=state.source_config,
            tool_history=[record.to_dict() for record in state.tool_history],
            observations=state.observations,
        )

        try:
            raw_output = model_client.generate(system_prompt, user_prompt, mode="tool_selection")
        except Exception as exc:  # noqa: BLE001
            state.fail(f"Model generation error: {exc}")
            logger.error("Model generation error on step %s: %s", state.current_step, exc)
            break

        try:
            action_payload = parse_model_action(raw_output)
        except ModelOutputParseError as exc:
            observation = f"PARSE_ERROR: {exc}"
            state.add_observation(observation)
            logger.warning("Malformed model output on step %s: %s", state.current_step, exc)
            state.advance_step()
            continue

        action = action_payload["action"]
        arguments = action_payload["arguments"]

        try:
            result = execute_action(registry, action, arguments)
        except (ToolNotFoundError, ToolDisabledError, InvalidArgumentsError) as exc:
            observation = f"ACTION_ERROR ({type(exc).__name__}): {exc}"
            state.add_observation(observation)
            state.errors.append(observation)
            logger.warning("Action error on step %s: %s", state.current_step, exc)
            state.advance_step()
            continue

        duration = float(result.get("duration_seconds", 0.0)) if isinstance(result, dict) else 0.0
        state.record_tool_call(action=action, arguments=arguments, result=result, duration_seconds=duration)

        error_message = None if result.get("success", True) else result.get("message")
        log_tool_call(
            source_id=source_id,
            step=state.current_step,
            tool_name=action,
            arguments=arguments,
            result=result,
            duration_seconds=duration,
            error=error_message,
        )

        state.add_observation(f"step={state.current_step} action={action} result={result}")

        if action == FINISH_ACTION:
            state.finish(reason=result.get("reason", "Model requested finish."))
            break

        state.advance_step()

    return state
