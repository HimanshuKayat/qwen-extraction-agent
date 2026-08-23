"""
Prompt construction for the autonomous extraction agent.

This module is responsible only for building messages sent to the model.

It does NOT:
- execute tools
- parse model JSON
- access files
- perform extraction
"""

from __future__ import annotations

import json
from typing import Any


def build_tool_selection_messages(
    source_config: dict[str, Any],
    tool_descriptions: list[dict[str, Any]],
    tool_history: list[dict[str, Any]] | None = None,
    observations: list[Any] | None = None,
) -> list[dict[str, str]]:
    """
    Build the messages for one tool-selection turn.

    The model receives:
        1. Its role and behavioral constraints.
        2. The source configuration.
        3. The currently available tools.
        4. Previous tool executions.
        5. Observations/results from previous steps.

    The model must return exactly one JSON action.
    """

    if tool_history is None:
        tool_history = []

    if observations is None:
        observations = []

    system_prompt = """
You are the autonomous decision-making agent for a data extraction system.

Your job is to choose the SINGLE best next action required to extract,
inspect, or validate data from the provided source.

You are the reasoning and decision layer.

You do NOT execute Python.
You do NOT execute shell commands.
You do NOT directly manipulate files.
You do NOT invent tools.
You do NOT call tools that are not listed.
You do NOT generate or recreate large datasets yourself.

Instead, choose exactly ONE controlled deterministic tool.

The tool execution layer will execute your selected action and return
the result to you on the next turn.

IMPORTANT OUTPUT RULES:

Return ONLY valid JSON.

Do NOT return:
- <think>
- </think>
- markdown
- code fences
- explanations
- comments
- natural-language text before the JSON
- natural-language text after the JSON

Your response MUST have exactly this structure:

{
  "action": "tool_name",
  "arguments": {}
}

IMPORTANT ARGUMENT RULES:

All tool arguments must contain plain JSON values.

URLs MUST be raw URLs.

Correct:
"https://example.com/"

Incorrect:
"[https://example.com/](https://example.com/)"

Incorrect:
"<https://example.com/>"

Do NOT use Markdown formatting inside tool arguments.

Do NOT add explanatory text to URLs, filenames, paths, or other
machine-readable arguments.

Use exactly the argument names and types defined by the selected tool.

Choose exactly ONE action per turn.

When the extraction process is complete, use:

{
  "action": "finish",
  "arguments": {
    "reason": "short explanation"
  }
}

The finish reason should be plain text.
""".strip()

    source_json = json.dumps(
        source_config,
        indent=2,
        ensure_ascii=False,
    )

    tools_json = json.dumps(
        tool_descriptions,
        indent=2,
        ensure_ascii=False,
    )

    history_json = json.dumps(
        tool_history,
        indent=2,
        ensure_ascii=False,
    )

    observations_json = json.dumps(
        observations,
        indent=2,
        ensure_ascii=False,
    )

    user_prompt = f"""
SOURCE CONFIGURATION:

{source_json}


AVAILABLE TOOLS:

{tools_json}


PREVIOUS TOOL CALLS:

{history_json}


OBSERVATIONS FROM PREVIOUS STEPS:

{observations_json}


Choose the single best next action.

Remember:
- Return ONLY JSON.
- Use raw URLs without Markdown.
- Use only arguments accepted by the selected tool.
- Choose exactly ONE action.

Return ONLY valid JSON.
""".strip()

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
