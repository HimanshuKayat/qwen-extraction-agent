import json

from agent.model import QwenModel
from agent.parser import parse_action
from agent.prompts import build_tool_selection_messages
from tools.registry import build_registry


def test_qwen_selects_http_download():
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

    assert action["action"] == "http_download"

    assert "url" in action["arguments"]
    assert "save_path" in action["arguments"]

    assert (
        action["arguments"]["url"]
        == "https://httpbin.org/bytes/100"
    )