"""Single-agent control loop.

SOURCE CONFIG
    ↓
Qwen
    ↓
JSON action
    ↓
controlled tool execution
    ↓
tool result
    ↓
Qwen
    ↓
next action
    ↓
...
    ↓
finish

The model is the only decision maker.
All execution is performed by deterministic tools.
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

from tools.registry import (
    FINISH_ACTION,
    ToolRegistry,
    execute_action,
)


logger = get_logger("agent.loop")


def run_agent(
    source_config: dict[str, Any],
    registry: ToolRegistry,
    model_client: ModelClient,
    max_steps: int = 20,
) -> AgentState:
    """
    Run the autonomous extraction agent for one source.

    The loop repeatedly:

    1. Builds the current agent context.
    2. Asks Qwen for exactly one action.
    3. Parses the model response.
    4. Executes the controlled tool.
    5. Records the result.
    6. Gives the result back to Qwen.
    7. Continues until finish/failure/max_steps.
    """

    state = AgentState(
        source_config=source_config,
        max_steps=max_steps,
    )

    source_id = source_config.get(
        "source_id",
        "unknown_source",
    )

    while state.is_active():

        # -----------------------------------------------------
        # BUILD CURRENT MODEL CONTEXT
        # -----------------------------------------------------

        tool_descriptions = registry.to_prompt_list()

        messages = build_tool_selection_messages(
            source_config=state.source_config,
            tool_descriptions=tool_descriptions,
            tool_history=[
                record.to_dict()
                for record in state.tool_history
            ],
            observations=state.observations,
        )

        # -----------------------------------------------------
        # ASK MODEL
        # -----------------------------------------------------

        try:

            raw_output = model_client.generate(
                messages,
                mode="tool_selection",
            )

        except Exception as exc:

            state.fail(
                f"Model generation error: {exc}"
            )

            logger.exception(
                "Model generation error on step %s",
                state.current_step,
            )

            break

        # -----------------------------------------------------
        # PARSE MODEL ACTION
        # -----------------------------------------------------

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

            logger.warning(
                "Malformed model output on step %s: %s",
                state.current_step,
                exc,
            )

            state.advance_step()

            continue

        # -----------------------------------------------------
        # EXTRACT ACTION
        # -----------------------------------------------------

        action = action_payload["action"]

        arguments = action_payload["arguments"]

        # -----------------------------------------------------
        # EXECUTE CONTROLLED TOOL
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # RECORD TOOL EXECUTION
        # -----------------------------------------------------

        duration = 0.0

        if isinstance(result, dict):

            duration = float(
                result.get(
                    "duration_seconds",
                    0.0,
                )
            )

        state.record_tool_call(
            action=action,
            arguments=arguments,
            result=result,
            duration_seconds=duration,
        )

        error_message = None

        if isinstance(result, dict):

            if not result.get(
                "success",
                True,
            ):

                error_message = result.get(
                    "message"
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

        # -----------------------------------------------------
        # ADD RESULT TO MODEL OBSERVATIONS
        # -----------------------------------------------------

        state.add_observation(
            {
                "step": state.current_step,
                "action": action,
                "result": result,
            }
        )

        # -----------------------------------------------------
        # FINISH
        # -----------------------------------------------------

        if action == FINISH_ACTION:

            reason = (
                result.get(
                    "reason",
                    "Model requested finish.",
                )
                if isinstance(result, dict)
                else "Model requested finish."
            )

            state.finish(
                reason=reason
            )

            break

        # -----------------------------------------------------
        # NEXT AGENT STEP
        # -----------------------------------------------------

        state.advance_step()

    return state
