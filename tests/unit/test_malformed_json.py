"""Unit tests for the model action parser."""

import pytest

from agent.parser import AgentResponseError, parse_action


def test_parse_valid_action():
    raw = (
        '{"action": "http_download", '
        '"arguments": {'
        '"url": "https://x", '
        '"save_path": "y"'
        '}}'
    )

    parsed = parse_action(raw)

    assert parsed["action"] == "http_download"
    assert parsed["arguments"]["url"] == "https://x"
    assert parsed["arguments"]["save_path"] == "y"


def test_parse_missing_action_key_raises():
    raw = '{"arguments": {}}'

    with pytest.raises(AgentResponseError):
        parse_action(raw)


def test_parse_non_json_raises():
    raw = "I will now call http_download with the url."

    with pytest.raises(AgentResponseError):
        parse_action(raw)


def test_parse_json_array_raises():
    raw = '[{"action": "finish", "arguments": {}}]'

    with pytest.raises(AgentResponseError):
        parse_action(raw)


def test_parse_action_wrong_type_raises():
    raw = '{"action": 123, "arguments": {}}'

    with pytest.raises(AgentResponseError):
        parse_action(raw)


def test_parse_arguments_wrong_type_raises():
    raw = '{"action": "finish", "arguments": "not-a-dict"}'

    with pytest.raises(AgentResponseError):
        parse_action(raw)


def test_parse_missing_arguments_raises():
    raw = '{"action": "finish"}'

    with pytest.raises(AgentResponseError):
        parse_action(raw)


def test_parse_markdown_fence_is_rejected():
    raw = (
        "```json\n"
        '{"action": "finish", "arguments": {"reason": "ok"}}'
        "\n```"
    )

    with pytest.raises(AgentResponseError):
        parse_action(raw)


def test_parse_empty_output_raises():
    raw = ""

    with pytest.raises(AgentResponseError):
        parse_action(raw)


def test_parse_invalid_json_raises():
    raw = '{"action": "finish", "arguments": '

    with pytest.raises(AgentResponseError):
        parse_action(raw)


def test_parse_extra_text_around_json_raises():
    raw = (
        'Here is the action: '
        '{"action": "finish", "arguments": {}}'
    )

    with pytest.raises(AgentResponseError):
        parse_action(raw)
