"""Integration test for Qwen tool selection.

Verifies that Qwen selects the correct HTTP download tool and produces
arguments matching the current raw-storage contract.
"""

from __future__ import annotations

import json

import pytest

from agent.model import QwenModel
from agent.parser import parse_action
from agent.prompts import build_tool_selection_messages
from tools.definitions import build_registry


@pytest.mark.integration
def test_qwen_selects_http_download():
    """Qwen should select http_download for a direct-download source."""

    model = QwenModel()

    registry = build_registry()

    tools = registry.to_prompt_list()

    source_config = {
        "source_id": "test_direct_download",
        "source_type": "1a",
        "title": "Test Direct Download",
        "data_link": "https://httpbin.org/bytes/100",
        "raw_dir": "raw/test_direct_download",
        "target_schema": {},
    }

    messages = build_tool_selection_messages(
        source_config=source_config,
        tool_descriptions=tools,
        tool_history=[],
        observations=[],
    )

    raw_response = model.generate(
        messages,
        mode="tool_selection",
    )

    print("\nRAW MODEL RESPONSE:")
    print(raw_response)

    action = parse_action(raw_response)

    print("\nPARSED ACTION:")
    print(json.dumps(action, indent=2))

    # ---------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------

    assert action["action"] == "http_download"

    arguments = action["arguments"]

    # ---------------------------------------------------------
    # Required download arguments
    # ---------------------------------------------------------

    assert "url" in arguments
    assert "source_id" in arguments
    assert "filename" in arguments

    # ---------------------------------------------------------
    # Validate values
    # ---------------------------------------------------------

    assert arguments["url"] == source_config["data_link"]

    assert arguments["source_id"] == source_config["source_id"]

    assert isinstance(arguments["filename"], str)
    assert arguments["filename"].strip() != ""

    # ---------------------------------------------------------
    # New storage contract
    # ---------------------------------------------------------

    # The model must NOT construct an arbitrary save_path.
    assert "save_path" not in arguments
