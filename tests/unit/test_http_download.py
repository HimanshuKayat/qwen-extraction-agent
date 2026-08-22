"""Unit tests for tools.http_tools.http_download.

Uses httpbin.org's /bytes endpoint as a live, small, deterministic test
target, matching the proving endpoint used elsewhere in the project.
"""

from pathlib import Path

import pytest

from core.exceptions import ToolExecutionError
from tools.http_tools import http_download


@pytest.mark.network
def test_http_download_success(tmp_path: Path):
    save_path = tmp_path / "test_download" / "sample.bin"
    result = http_download(url="https://httpbin.org/bytes/100", save_path=str(save_path))

    assert result["success"] is True
    assert result["bytes"] == 100
    assert Path(result["file_path"]).exists()
    assert Path(result["file_path"]).stat().st_size == 100


@pytest.mark.network
def test_http_download_404_raises_tool_execution_error(tmp_path: Path):
    save_path = tmp_path / "test_download" / "missing.bin"
    with pytest.raises(ToolExecutionError):
        http_download(url="https://httpbin.org/status/404", save_path=str(save_path))


def test_http_download_creates_parent_directories(tmp_path: Path, monkeypatch):
    """Verify parent-directory creation logic without a real network call."""

    class FakeResponse:
        content = b"hello"
        headers = {"content-type": "text/plain"}

        def raise_for_status(self):
            return None

    def fake_get(url, timeout, allow_redirects):
        return FakeResponse()

    monkeypatch.setattr("tools.http_tools.requests.get", fake_get)

    save_path = tmp_path / "nested" / "dir" / "file.txt"
    result = http_download(url="https://example.com/file.txt", save_path=str(save_path))

    assert result["success"] is True
    assert save_path.exists()
    assert save_path.read_bytes() == b"hello"
