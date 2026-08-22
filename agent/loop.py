"""
The autonomous agent control loop.

SOURCE CONFIG
    ↓
Qwen / model client
    ↓
JSON action
    ↓
parse_action
    ↓
controlled ToolRegistry
    ↓
tool execution
    ↓
observation
    ↓
model again
    ↓
...
    ↓
finish

This is a SINGLE agent.

The model is the decision maker.
Tools are deterministic execution components.
The model never directly executes Python, shell commands, or arbitrary code.
"""

from __future__ import annotations

from typing import Any

from agent.model import ModelClient
from agent.parser import AgentResponseError, parse_action
from agent.prompts import build_tool_selection_messages
from agent.state import AgentState
from core.exceptions import (
    InvalidArgumentsError,
    ToolDisabledError,
    ToolNotFoundError,
)
from core.logging_utils import get_logger, log_tool_call
from tools.registry import FINISH_ACTION, ToolRegistry, execute_action


logger = get_logger("agent.loop")


def run_agent(
    source_config: dict[str, Any],
    registry: ToolRegistry,
    model_client: ModelClient,
    max_steps: int = 20,
) -> AgentState:
    """
    Run the autonomous extraction agent for one source.

    Each step follows:

        1. Build the current model prompt.
        2. Ask the model for exactly one JSON action.
        3. Parse and validate the action.
        4. Execute it through the controlled registry.
        5. Store the result as an observation.
        6. Give the updated state back to the model.
        7. Continue until finish, failure, or max_steps.

    Args:
        source_config:
            Source configuration dictionary.

        registry:
            Controlled ToolRegistry containing executable tools.

        model_client:
            Model client implementing:

                generate(
                    system_prompt,
                    user_prompt,
                    mode="tool_selection",
                )

        max_steps:
            Maximum number of model/tool turns.

    Returns:
        Final AgentState.
    """

    state = AgentState(
        source_config=source_config,
        max_steps=max_steps,
    )

    source_id = source_config.get(
        "source_id",
        "unknown_source",
    )

    tool_descriptions = registry.to_prompt_list()

    while state.is_active():

        # -----------------------------------------------------------
        # 1. Build current prompt
        # -----------------------------------------------------------

        messages = build_tool_selection_messages(
            source_config=state.source_config,
            tool_descriptions=tool_descriptions,
            tool_history=[
                record.to_dict()
                for record in state.tool_history
            ],
            observations=state.observations,
        )

        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]

        # -----------------------------------------------------------
        # 2. Ask model for next action
        # -----------------------------------------------------------

        try:
            raw_output = model_client.generate(
                system_prompt,
                user_prompt,
                mode="tool_selection",
            )

        except Exception as exc:  # noqa: BLE001
            state.fail(
                f"Model generation error: {exc}"
            )

            logger.exception(
                "Model generation error on step %s",
                state.current_step,
            )

            break

        # -----------------------------------------------------------
        # 3. Parse model action
        # -----------------------------------------------------------

        try:
            action_payload = parse_action(
                raw_output
            )

        except AgentResponseError as exc:
            observation = (
                f"PARSE_ERROR: {exc}"
            )

            state.add_observation(
                observation
            )

            state.errors.append(
                observation
            )

            logger.warning(
                "Malformed model output on step %s: %s",
                state.current_step,
                exc,
            )

            state.advance_step()

            continue

        # -----------------------------------------------------------
        # 4. Extract action
        # -----------------------------------------------------------

        action = action_payload["action"]
        arguments = action_payload["arguments"]

        logger.info(
            "Model selected action on step %s: %s",
            state.current_step,
            action,
        )

        # -----------------------------------------------------------
        # 5. Execute through controlled registry
        # -----------------------------------------------------------

        try:
            result = execute_action(
                registry,
                action,
                arguments,
            )

        except (
            ToolNotFoundError,
            ToolDisabledError,
            InvalidArgumentsError,
        ) as exc:

            observation = (
                f"ACTION_ERROR "
                f"({type(exc).__name__}): {exc}"
            )

            state.add_observation(
                observation
            )

            state.errors.append(
                observation
            )

            logger.warning(
                "Action error on step %s: %s",
                state.current_step,
                exc,
            )

            state.advance_step()

            continue

        # -----------------------------------------------------------
        # 6. Record tool execution
        # -----------------------------------------------------------

        duration = (
            float(
                result.get(
                    "duration_seconds",
                    0.0,
                )
            )
            if isinstance(result, dict)
            else 0.0
        )

        state.record_tool_call(
            action=action,
            arguments=arguments,
            result=result,
            duration_seconds=duration,
        )

        error_message = (
            None
            if result.get("success", True)
            else result.get("message")
        )

        log_tool_call(
            source_id=source_id,
            step=state.current_step,
            tool_name=action,
            arguments=arguments,
            result=result,
            duration_seconds=duration,
            error=error_message,
        )

        # -----------------------------------------------------------
        # 7. Add result to model's next observation
        # -----------------------------------------------------------

        state.add_observation(
            f"step={state.current_step} "
            f"action={action} "
            f"result={result}"
        )

        # -----------------------------------------------------------
        # 8. Finish condition
        # -----------------------------------------------------------

        if action == FINISH_ACTION:

            state.finish(
                reason=result.get(
                    "reason",
                    "Model requested finish.",
                )
            )

            break

        # -----------------------------------------------------------
        # 9. Continue to next model turn
        # -----------------------------------------------------------

        state.advance_step()

    return state
