"""End-to-end Qwen + NPCI website discovery test.

Flow:

    NPCI URL
       ↓
    Qwen
       ↓
    browser_open
       ↓
    browser_inspect
       ↓
    page observation
       ↓
    Qwen decides the next action

The test intentionally does NOT provide a direct dataset URL.
"""

from __future__ import annotations

import json

from agent.model import QwenModel
from agent.parser import parse_action
from agent.prompts import build_tool_selection_messages
from tools.definitions import build_registry
from tools.registry import execute_action


NPCI_URL = "https://www.npci.org.in/product/upi/product-statistics"


def test_qwen_npci_discovery():
    """Qwen should open and inspect the real NPCI UPI statistics page."""

    registry = build_registry()

    model = QwenModel()

    source_config = {
        "source_id": "npci_upi_daily",
        "source_type": "webpage",
        "title": "NPCI UPI Daily Statistics",
        "data_link": NPCI_URL,
        "description": (
            "Open the official NPCI UPI Product Statistics page. "
            "Inspect the page and identify the Daily Statistics "
            "data or a downloadable resource containing daily UPI "
            "transaction statistics. Do not assume or invent a "
            "download URL."
        ),
        "target_schema": {
            "frequency": "daily",
            "subject": "UPI transactions",
            "fields": [
                "date",
                "transaction_volume",
                "transaction_value",
            ],
        },
    }

    # ---------------------------------------------------------
    # STEP 1 — Ask Qwen what to do
    # ---------------------------------------------------------

    print("\n[1/5] Building Qwen prompt...")

    messages = build_tool_selection_messages(
        source_config=source_config,
        tool_descriptions=registry.to_prompt_list(),
        tool_history=[],
        observations=[],
    )

    print("[2/5] Asking Qwen for first action...")

    raw_output = model.generate(
        messages,
        mode="tool_selection",
    )

    print("\nRAW QWEN RESPONSE:")
    print(raw_output)

    action = parse_action(raw_output)

    print("\nPARSED ACTION:")
    print(json.dumps(action, indent=2))

    assert action["action"] == "browser_open"

    # ---------------------------------------------------------
    # STEP 2 — Execute browser_open
    # ---------------------------------------------------------

    print("\n[3/5] Executing browser_open...")

    open_result = execute_action(
        registry,
        action["action"],
        action["arguments"],
    )

    print("\nBROWSER OPEN RESULT:")
    print(open_result)

    assert open_result["success"] is True
    assert open_result["status_code"] == 200

    # ---------------------------------------------------------
    # STEP 3 — Inspect the actual NPCI page
    # ---------------------------------------------------------

    print("\n[4/5] Inspecting NPCI page...")

    inspect_result = execute_action(
        registry,
        "browser_inspect",
        {},
    )

    print("\nNPCI PAGE INSPECTION:")
    print(json.dumps(inspect_result, indent=2, ensure_ascii=False))

    assert inspect_result["success"] is True

    # ---------------------------------------------------------
    # STEP 4 — Give the observation back to Qwen
    # ---------------------------------------------------------

    print("\n[5/5] Asking Qwen what to do next...")

    tool_history = [
        {
            "step": 0,
            "action": action["action"],
            "arguments": action["arguments"],
            "result": open_result,
        }
    ]

    observations = [
        {
            "step": 0,
            "tool": "browser_inspect",
            "result": inspect_result,
        }
    ]

    messages = build_tool_selection_messages(
        source_config=source_config,
        tool_descriptions=registry.to_prompt_list(),
        tool_history=tool_history,
        observations=observations,
    )

    raw_next_output = model.generate(
        messages,
        mode="tool_selection",
    )

    print("\nRAW QWEN NEXT RESPONSE:")
    print(raw_next_output)

    next_action = parse_action(raw_next_output)

    print("\nPARSED NEXT ACTION:")
    print(json.dumps(next_action, indent=2))

    # We are not forcing a particular second action yet.
    #
    # The important thing is that Qwen has now received the
    # actual NPCI webpage observation and must decide what
    # controlled action is appropriate next.

    assert next_action["action"] in {
        "browser_open",
        "browser_inspect",
        "browser_close",
        "http_download",
        "inspect_file",
        "read_csv",
        "read_excel",
        "read_pdf",
        "extract_pdf_table",
        "validate_required_fields",
        "validate_row_count",
        "finish",
    }

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    try:
        execute_action(
            registry,
            "browser_close",
            {},
        )
    except Exception:
        pass
