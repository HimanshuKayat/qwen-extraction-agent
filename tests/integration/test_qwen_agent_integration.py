"""Integration test for the full agent loop:

SOURCE CONFIG -> model -> JSON action -> execute_action -> tool result
-> model -> next action -> finish

This test does NOT require a live Qwen model or GPU. It uses a small
scripted ModelClient double that mimics correct tool-selection output
for the proving source (https://httpbin.org/bytes/100), so the loop's
plumbing can be verified in any environment, including CI.

A real end-to-end run against the actual Qwen3-8B model is a separate,
GPU-requiring exercise driven from notebooks/01_qwen_agent_test.ipynb,
not from this automated test suite.
"""

import json
from pathlib import Path

import pytest

from agent.loop import run_agent
from agent.state import AgentStatus
from core.config_loader import load_source_config
from tools.definitions import build_registry

REPO_ROOT = Path(__file__).resolve().parents[2]


class ScriptedModelClient:
    """A ModelClient double that returns pre-scripted JSON responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, mode: str = "tool_selection") -> str:
        response = self._responses[self.calls]
        self.calls += 1
        return response


@pytest.mark.network
def test_full_agent_loop_downloads_and_finishes(tmp_path: Path):
    source_config = load_source_config(REPO_ROOT / "config" / "sources" / "test_direct_download.yaml")
    save_path = str(tmp_path / "test_direct_download" / "sample.bin")

    scripted_responses = [
        json.dumps({
            "action": "http_download",
            "arguments": {"url": source_config["data_link"], "save_path": save_path},
        }),
        json.dumps({
            "action": "finish",
            "arguments": {"reason": "File downloaded successfully."},
        }),
    ]

    registry = build_registry()
    model_client = ScriptedModelClient(scripted_responses)

    final_state = run_agent(source_config, registry, model_client, max_steps=10)

    assert final_state.status == AgentStatus.FINISHED
    assert len(final_state.tool_history) == 2
    assert final_state.tool_history[0].action == "http_download"
    assert final_state.tool_history[0].result["success"] is True
    assert Path(save_path).exists()
    assert model_client.calls == 2


def test_agent_loop_handles_malformed_json_then_recovers():
    source_config = {
        "source_id": "malformed_test",
        "source_type": "1a",
        "title": "Malformed Test",
        "data_link": "https://example.com/does-not-matter",
        "raw_dir": "raw/malformed_test",
        "target_schema": {},
    }

    scripted_responses = [
        "this is not json at all",
        json.dumps({"action": "finish", "arguments": {"reason": "recovered"}}),
    ]

    registry = build_registry()
    model_client = ScriptedModelClient(scripted_responses)

    final_state = run_agent(source_config, registry, model_client, max_steps=10)

    assert final_state.status == AgentStatus.FINISHED
    assert any("PARSE_ERROR" in obs for obs in final_state.observations)


def test_agent_loop_stops_at_max_steps_if_model_never_finishes():
    source_config = {
        "source_id": "never_finishes",
        "source_type": "1a",
        "title": "Never Finishes",
        "data_link": "https://example.com/does-not-matter",
        "raw_dir": "raw/never_finishes",
        "target_schema": {},
    }

    # Always request an unknown tool, so the loop keeps recording errors
    # and stepping forward without ever finishing.
    scripted_responses = [json.dumps({"action": "unknown_tool", "arguments": {}})] * 10

    registry = build_registry()
    model_client = ScriptedModelClient(scripted_responses)

    final_state = run_agent(source_config, registry, model_client, max_steps=3)

    assert final_state.status == AgentStatus.MAX_STEPS_REACHED
