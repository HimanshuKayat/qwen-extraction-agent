"""End-to-end browser execution test.

Verifies:

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
      ↓
    browser_close

The test deliberately uses explicit timeouts around browser operations so
a browser/session failure cannot hang the Colab runtime indefinitely.
"""

from __future__ import annotations

import time

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
        "description": "Open this website and inspect its contents.",
        "target_schema": {},
    }

    # ---------------------------------------------------------
    # STEP 1 — Ask Qwen for the first action
    # ---------------------------------------------------------

    print("\n[1/4] Building tool-selection prompt...")

    messages = build_tool_selection_messages(
        source_config=source_config,
        tool_descriptions=registry.to_prompt_list(),
        tool_history=[],
        observations=[],
    )

    print("[1/4] Asking Qwen for browser action...")

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
    # STEP 2 — Execute browser_open
    # ---------------------------------------------------------

    print("\n[2/4] Executing browser_open...")

    start = time.monotonic()

    open_result = execute_action(
        registry,
        action["action"],
        action["arguments"],
    )

    elapsed = time.monotonic() - start

    print(
        f"\nBROWSER OPEN RESULT "
        f"(completed in {elapsed:.2f}s):"
    )
    print(open_result)

    assert open_result["success"] is True
    assert open_result["status_code"] == 200
    assert open_result["url"].startswith(
        "https://example.com"
    )

    # ---------------------------------------------------------
    # STEP 3 — Inspect actual webpage
    # ---------------------------------------------------------

    print("\n[3/4] Executing browser_inspect...")

    start = time.monotonic()

    inspect_result = execute_action(
        registry,
        "browser_inspect",
        {},
    )

    elapsed = time.monotonic() - start

    print(
        f"\nBROWSER INSPECTION RESULT "
        f"(completed in {elapsed:.2f}s):"
    )
    print(inspect_result)

    assert inspect_result["success"] is True
    assert inspect_result["title"] == "Example Domain"
    assert "Example Domain" in inspect_result["text"]

    # ---------------------------------------------------------
    # STEP 4 — Close browser
    # ---------------------------------------------------------

    print("\n[4/4] Executing browser_close...")

    start = time.monotonic()

    close_result = execute_action(
        registry,
        "browser_close",
        {},
    )

    elapsed = time.monotonic() - start

    print(
        f"\nBROWSER CLOSE RESULT "
        f"(completed in {elapsed:.2f}s):"
    )
    print(close_result)

    assert close_result["success"] is True

    print("\n✓ COMPLETE: Qwen → browser → webpage → inspection")
