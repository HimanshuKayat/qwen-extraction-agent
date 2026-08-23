"""End-to-end browser execution test.

Verifies that:

    Qwen
      ↓
    browser_open
      ↓
    Playwright
      ↓
    real webpage
      ↓
    browser_inspect
      ↓
    actual page content

The model makes the browser-open decision. The registry executes it.
"""

from __future__ import annotations

from agent.model import QwenModel
from agent.parser import parse_action
from agent.prompts import build_tool_selection_messages
from tools.definitions import build_registry
from tools.registry import execute_action


def test_qwen_browser_open_and_inspect():
    """Qwen selects browser_open and the browser actually executes it."""

    registry = build_registry()

    model = QwenModel()

    source_config = {
        "source_id": "example_website",
        "source_type": "webpage",
        "title": "Example Website",
        "data_link": "https://example.com/",
        "description": (
            "Open this website and inspect its contents."
        ),
        "target_schema": {},
    }

    # ---------------------------------------------------------
    # STEP 1 — Ask Qwen for the first action
    # ---------------------------------------------------------

    messages = build_tool_selection_messages(
        source_config=source_config,
        tool_descriptions=registry.to_prompt_list(),
        tool_history=[],
        observations=[],
    )

    raw_output = model.generate(
        messages,
        mode="tool_selection",
    )

    print("\nRAW QWEN RESPONSE:")
    print(raw_output)

    action = parse_action(raw_output)

    print("\nPARSED ACTION:")
    print(action)

    assert action["action"] == "browser_open"

    # ---------------------------------------------------------
    # STEP 2 — Execute Qwen's selected action
    # ---------------------------------------------------------

    open_result = execute_action(
        registry,
        action["action"],
        action["arguments"],
    )

    print("\nBROWSER OPEN RESULT:")
    print(open_result)

    assert open_result["success"] is True
    assert open_result["status_code"] == 200
    assert open_result["url"].startswith(
        "https://example.com"
    )

    # ---------------------------------------------------------
    # STEP 3 — Inspect the actual webpage
    # ---------------------------------------------------------

    inspect_result = execute_action(
        registry,
        "browser_inspect",
        {},
    )

    print("\nBROWSER INSPECTION RESULT:")
    print(inspect_result)

    assert inspect_result["success"] is True
    assert inspect_result["title"] == "Example Domain"
    assert "Example Domain" in inspect_result["text"]

    # ---------------------------------------------------------
    # STEP 4 — Close browser
    # ---------------------------------------------------------

    close_result = execute_action(
        registry,
        "browser_close",
        {},
    )

    print("\nBROWSER CLOSE RESULT:")
    print(close_result)

    assert close_result["success"] is True
