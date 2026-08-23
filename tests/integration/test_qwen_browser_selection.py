"""Integration test for Qwen browser tool selection.

The model receives a webpage source and should select browser_open
as its first action.

This verifies:

    source configuration
        ↓
    tool descriptions
        ↓
    Qwen
        ↓
    JSON response
        ↓
    action parser

It does not execute the browser.
"""

from __future__ import annotations

import json

import pytest

from agent.model import QwenModel
from agent.parser import parse_action
from agent.prompts import build_prompt
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

    # Use the current prompt-building API.
    prompt = build_prompt(
        source_config=source_config,
        tools=registry.to_prompt_list(),
        tool_history=[],
        observations=[],
    )

    raw_output = model.generate(
        prompt,
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
        )
    )

    assert parsed["action"] == "browser_open"

    assert parsed["arguments"]["url"] == (
        "https://example.com/"
    )
