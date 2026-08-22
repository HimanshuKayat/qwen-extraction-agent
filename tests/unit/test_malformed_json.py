"""Unit tests for agent.prompts.parse_model_action: the model output
contract must be strictly enforced, and malformed output must never be
silently executed.
"""

import pytest

from agent.prompts import parse_model_action
from core.exceptions import ModelOutputParseError


def test_parse_valid_action():
    raw = '{"action": "http_download", "arguments": {"url": "https://x", "save_path": "y"}}'
    parsed = parse_model_action(raw)
    assert parsed["action"] == "http_download"
    assert parsed["arguments"]["url"] == "https://x"


def test_parse_strips_markdown_fence():
    raw = '```json\n{"action": "finish", "arguments": {"reason": "ok"}}\n```'
    parsed = parse_model_action(raw)
    assert parsed["action"] == "finish"


def test_parse_missing_action_key_raises():
    raw = '{"arguments": {}}'
    with pytest.raises(ModelOutputParseError):
        parse_model_action(raw)


def test_parse_non_json_raises():
    raw = "I will now call http_download with the url."
    with pytest.raises(ModelOutputParseError):
        parse_model_action(raw)


def test_parse_json_array_raises():
    raw = '[{"action": "finish", "arguments": {}}]'
    with pytest.raises(ModelOutputParseError):
        parse_model_action(raw)


def test_parse_action_wrong_type_raises():
    raw = '{"action": 123, "arguments": {}}'
    with pytest.raises(ModelOutputParseError):
        parse_model_action(raw)


def test_parse_arguments_wrong_type_raises():
    raw = '{"action": "finish", "arguments": "not-a-dict"}'
    with pytest.raises(ModelOutputParseError):
        parse_model_action(raw)


def test_parse_defaults_missing_arguments_to_empty_dict():
    raw = '{"action": "finish"}'
    parsed = parse_model_action(raw)
    assert parsed["arguments"] == {}
