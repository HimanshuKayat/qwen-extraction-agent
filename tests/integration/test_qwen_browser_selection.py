"""Integration test for Qwen browser tool selection.

The model receives a webpage source and must select browser_open
as its first action.

This test verifies model -> JSON action parsing, not actual browser
execution.
"""

import json

import pytest

from agent.model import QwenModel
from agent.prompts import build_system_prompt, build_user_prompt, parse_model_action
from tools.definitions import build_registry


@pytest.mark.integration
def test_qwen_selects_browser_open():
    """Qwen should select browser_open for a webpage source."""

    registry = build_registry()

    model = QwenModel()

    source_config = {
        "source_id": "example_website",
        "source_type": "webpage",
        "title": "Example Website",
        "data_link": "https://example.com/",
        "description": (
            "Open this website and inspect it to determine "
            "whether it contains downloadable data."
        ),
        "target_schema": {},
    }

    system_prompt = build_system_prompt(
        registry.to_prompt_list()
    )

    user_prompt = build_user_prompt(
        source_config=source_config,
        tool_history=[],
        observations=[],
    )

    raw_output = model.generate(
        system_prompt,
        user_prompt,
        mode="tool_selection",
    )

    print("\nRAW MODEL RESPONSE:")
    print(raw_output)

    parsed = parse_model_action(raw_output)

    print("\nPARSED ACTION:")
    print(
        json.dumps(
            parsed,
            indent=2,
        )
    )

    assert parsed["action"] == "browser_open"

    assert parsed["arguments"]["url"] == (
        "https://example.com/"
    )
