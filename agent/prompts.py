from __future__ import annotations

import json
from typing import Any


def build_tool_selection_messages(
    source_config: dict[str, Any],
    tool_descriptions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Build the messages used when asking Qwen to select
    the next deterministic tool.
    """

    system_prompt = """
You are an autonomous data-extraction agent.

Your job is to select exactly ONE action from the
currently available tools.

You are the reasoning/controller layer.

You do NOT execute Python.
You do NOT execute shell commands.
You do NOT directly manipulate files.
You do NOT generate large datasets.

Instead, select a controlled deterministic tool.

IMPORTANT OUTPUT RULES:

1. Return ONLY valid JSON.
2. Do NOT return <think>.
3. Do NOT return markdown.
4. Do NOT return explanations.
5. Do NOT put text before or after the JSON.
6. Use ONLY tools listed below.
7. Use ONLY arguments defined by the selected tool.

Required format:

{
  "action": "tool_name",
  "arguments": {
    "argument_name": "value"
  }
}

If the extraction process is complete, use:

{
  "action": "finish",
  "arguments": {
    "reason": "..."
  }
}
""".strip()

    tools_json = json.dumps(
        tool_descriptions,
        indent=2,
        ensure_ascii=False,
    )

    source_json = json.dumps(
        source_config,
        indent=2,
        ensure_ascii=False,
    )

    user_prompt = f"""
AVAILABLE TOOLS:

{tools_json}

SOURCE CONFIGURATION:

{source_json}

Determine the single best next action.

Remember:

- Return ONLY JSON.
- Select exactly one available tool.
- Do not explain your decision.
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