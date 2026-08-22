"""Integration tests for the complete agent loop.

These tests verify:

    source config
        ↓
    model action
        ↓
    ToolRegistry
        ↓
    http_download
        ↓
    controlled raw storage
        ↓
    finish

The model is scripted here so the integration test does not require
loading Qwen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.loop import run_agent
from agent.state import AgentStatus
from core.config_loader import load_source_config
from storage.paths import raw_path
from tools.definitions import build_registry


REPO_ROOT = Path(__file__).resolve().parents[2]


class ScriptedModelClient:
    """Simple deterministic model client for integration tests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0

    def generate(
        self,
        system_prompt,
        user_prompt,
        mode="tool_selection",
    ):
        if self.index >= len(self.responses):
            raise RuntimeError(
                "ScriptedModelClient ran out of responses."
            )

        response = self.responses[self.index]
        self.index += 1

        return response


@pytest.mark.network
def test_full_agent_loop_downloads_and_finishes(
    tmp_path: Path,
    monkeypatch,
):
    """The agent should download a file and then finish successfully."""

    source_config = load_source_config(
        REPO_ROOT
        / "config"
        / "sources"
        / "test_direct_download.yaml"
    )

    source_id = source_config["source_id"]

    # Redirect application raw storage to pytest's temporary directory.
    monkeypatch.setattr(
        "storage.paths.STORAGE_ROOT",
        tmp_path / "storage",
    )
    monkeypatch.setattr(
        "storage.paths.RAW_ROOT",
        tmp_path / "storage" / "raw",
    )

    filename = "sample.bin"

    scripted_responses = [
        json.dumps(
            {
                "action": "http_download",
                "arguments": {
                    "url": source_config["data_link"],
                    "source_id": source_id,
                    "filename": filename,
                },
            }
        ),
        json.dumps(
            {
                "action": "finish",
                "arguments": {
                    "reason": "File downloaded successfully.",
                },
            }
        ),
    ]

    registry = build_registry()

    model_client = ScriptedModelClient(
        scripted_responses
    )

    final_state = run_agent(
        source_config,
        registry,
        model_client,
        max_steps=10,
    )

    assert final_state.status == AgentStatus.FINISHED

    expected_path = raw_path(
        source_id,
        filename,
    )

    assert expected_path.exists()
    assert expected_path.stat().st_size == 100

    assert len(final_state.tool_history) == 2

    assert (
        final_state.tool_history[0].action
        == "http_download"
    )

    assert (
        final_state.tool_history[1].action
        == "finish"
    )


def test_agent_loop_handles_malformed_json_then_recovers():
    """The agent should recover when the model first returns malformed JSON."""

    source_config = {
        "source_id": "malformed_json_test",
        "source_type": "1a",
        "data_link": "https://example.com/test.bin",
    }

    scripted_responses = [
        "this is not valid JSON",
        json.dumps(
            {
                "action": "finish",
                "arguments": {
                    "reason": "Recovered after malformed output.",
                },
            }
        ),
    ]

    registry = build_registry()

    model_client = ScriptedModelClient(
        scripted_responses
    )

    final_state = run_agent(
        source_config,
        registry,
        model_client,
        max_steps=10,
    )

    assert final_state.status == AgentStatus.FINISHED

    assert len(final_state.errors) == 0 or any(
        "PARSE_ERROR" in observation
        for observation in final_state.observations
    )


def test_agent_loop_stops_at_max_steps_if_model_never_finishes():
    """The agent must stop rather than loop forever."""

    source_config = {
        "source_id": "max_steps_test",
        "source_type": "1a",
        "data_link": "https://example.com/test.bin",
    }

    scripted_responses = [
        json.dumps(
            {
                "action": "unknown_tool",
                "arguments": {},
            }
        )
    ] * 10

    registry = build_registry()

    model_client = ScriptedModelClient(
        scripted_responses
    )

    max_steps = 3

    final_state = run_agent(
        source_config,
        registry,
        model_client,
        max_steps=max_steps,
    )

    assert final_state.current_step >= max_steps
    assert final_state.status != AgentStatus.FINISHED
