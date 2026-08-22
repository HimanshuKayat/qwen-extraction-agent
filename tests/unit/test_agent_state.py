"""Unit tests for agent.state.AgentState."""

from agent.state import AgentState, AgentStatus


def make_state(max_steps: int = 3) -> AgentState:
    return AgentState(source_config={"source_id": "test_source"}, max_steps=max_steps)


def test_initial_state_is_running():
    state = make_state()
    assert state.status == AgentStatus.RUNNING
    assert state.is_active() is True
    assert state.current_step == 0


def test_record_tool_call_appends_history_and_artifact():
    state = make_state()
    result = {"success": True, "file_path": "raw/test_source/file.bin", "bytes": 100}
    state.record_tool_call("http_download", {"url": "https://x"}, result, duration_seconds=0.1)

    assert len(state.tool_history) == 1
    assert state.tool_history[0].action == "http_download"
    assert "raw/test_source/file.bin" in state.artifacts


def test_record_failed_tool_call_appends_error():
    state = make_state()
    result = {"success": False, "message": "boom"}
    state.record_tool_call("http_download", {"url": "https://x"}, result, duration_seconds=0.1)

    assert any("boom" in error for error in state.errors)


def test_advance_step_reaches_max_steps():
    state = make_state(max_steps=2)
    state.advance_step()
    assert state.status == AgentStatus.RUNNING
    state.advance_step()
    assert state.status == AgentStatus.MAX_STEPS_REACHED
    assert state.is_active() is False


def test_finish_sets_status_and_reason():
    state = make_state()
    state.finish("File downloaded successfully.")
    assert state.status == AgentStatus.FINISHED
    assert state.finish_reason == "File downloaded successfully."
    assert state.is_active() is False


def test_fail_sets_status_reason_and_error():
    state = make_state()
    state.fail("Model unreachable.")
    assert state.status == AgentStatus.FAILED
    assert "Model unreachable." in state.errors


def test_to_dict_and_to_json_round_trip_shape():
    state = make_state()
    state.finish("done")
    as_dict = state.to_dict()
    assert as_dict["status"] == "finished"
    assert isinstance(state.to_json(), str)
