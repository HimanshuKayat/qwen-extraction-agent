"""Integration test for Qwen browser tool selection.

The model receives a webpage source and should select browser_open
as its first action.

This test verifies:

    source configuration
        ↓
    tool descriptions
        ↓
    Qwen
        ↓
    JSON response
        ↓
    action parser

The browser itself is NOT executed in this test.
"""

from __future__ import annotations

import json

import pytest

from agent.model import QwenModel
from agent.parser import parse_action
from agent.prompts import build_tool_selection_messages
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

    messages = build_tool_selection_messages(
        source_config=source_config,
        tool_descriptions=registry.to_prompt_list(),
        tool_history=[],
        observations=[],
    )

    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    raw_output = model.generate(
        system_prompt,
        user_prompt,
        mode="tool_selection",
    )

    print("\nRAW MODEL RESPONSE:")
    print(raw_output)

    parsed = parse_action(raw_output)

    print("\nPARSED ACTION:")
    print(
        json.dumps(
            parsed,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert parsed["action"] == "browser_open"

    assert parsed["arguments"]["url"] == (
        "https://example.com/"
    )
