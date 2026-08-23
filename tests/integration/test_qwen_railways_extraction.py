"""End-to-end Qwen selection test for an Indian Railways PDF.

Target:
Indian Railways Janshatabdi Express 2026 timetable.

The first goal is to verify that Qwen recognizes the source
as a downloadable PDF and selects the appropriate first tool.
"""

from __future__ import annotations

import json

from agent.model import QwenModel
from agent.parser import parse_action
from agent.prompts import build_tool_selection_messages
from tools.definitions import build_registry


RAILWAYS_URL = (
    "https://indianrailways.gov.in/"
    "railwayboard/uploads/directorate/coaching/"
    "TAG_2026/Janshatabdi_Exp.pdf"
)


def test_qwen_selects_railways_pdf_download():
    """Qwen should select http_download for the Railway PDF."""

    print("\n" + "=" * 70)
    print("INDIAN RAILWAYS — QWEN PDF SELECTION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 1 — Registry
    # ---------------------------------------------------------

    print("\n[1/4] Loading registry...")

    registry = build_registry()

    print("Available tools:")

    print([
        spec.name
        for spec in registry.list_enabled()
    ])

    # ---------------------------------------------------------
    # STEP 2 — Qwen
    # ---------------------------------------------------------

    print("\n[2/4] Loading Qwen...")

    model = QwenModel()

    # ---------------------------------------------------------
    # STEP 3 — Source configuration
    # ---------------------------------------------------------

    source_config = {
        "source_id": (
            "indian_railways_janshatabdi_2026"
        ),
        "source_type": "pdf",
        "title": (
            "Indian Railways Janshatabdi Express "
            "2026 Timetable"
        ),
        "data_link": RAILWAYS_URL,
        "description": (
            "Indian Railways 2026 Janshatabdi Express "
            "timetable containing train numbers, service "
            "days, origins, destinations, departure times "
            "and arrival times."
        ),
        "target_schema": {
            "train_number": "string",
            "days_of_service": "string",
            "origin": "string",
            "destination": "string",
            "departure_time": "string",
            "arrival_time": "string",
        },
    }

    print("\n[3/4] Building Qwen prompt...")

    messages = build_tool_selection_messages(
        source_config=source_config,
        tool_descriptions=registry.to_prompt_list(),
        tool_history=[],
        observations=[],
    )

    # ---------------------------------------------------------
    # STEP 4 — Ask Qwen
    # ---------------------------------------------------------

    print("[4/4] Asking Qwen for first action...")

    raw_response = model.generate(
        messages,
        mode="tool_selection",
    )

    print("\nRAW QWEN RESPONSE:")
    print(raw_response)

    action = parse_action(raw_response)

    print("\nPARSED ACTION:")
    print(
        json.dumps(
            action,
            indent=2,
            ensure_ascii=False,
        )
    )

    # ---------------------------------------------------------
    # Expected first action
    # ---------------------------------------------------------

    assert action["action"] == "http_download"

    assert "url" in action["arguments"]

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("Qwen correctly selected http_download.")
    print("=" * 70)
