"""Unit tests for tools.http_tools.http_download.

Network access is mocked so the unit tests remain deterministic and do not
depend on external services such as httpbin.org.
"""

from pathlib import Path

import pytest

from core.exceptions import ToolExecutionError
from tools.http_tools import http_download


class FakeResponse:
    """Minimal requests.Response replacement for tests."""

    def __init__(
        self,
        content: bytes = b"",
        content_type: str = "application/octet-stream",
        status_code: int = 200,
    ):
        self.content = content
        self.headers = {
            "content-type": content_type,
        }
        self.status_code = status_code
        self.url = "https://example.com/test"

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(
                f"{self.status_code} Server Error",
                response=self,
            )


def test_http_download_success(tmp_path: Path, monkeypatch):
    """A successful HTTP response is saved correctly."""

    def fake_get(url, timeout, allow_redirects):
        assert url == "https://example.com/test.bin"
        assert timeout == 60
        assert allow_redirects is True

        return FakeResponse(
            content=b"x" * 100,
            content_type="application/octet-stream",
        )

    monkeypatch.setattr(
        "tools.http_tools.requests.get",
        fake_get,
    )

    save_path = tmp_path / "test_download" / "sample.bin"

    result = http_download(
        url="https://example.com/test.bin",
        save_path=str(save_path),
    )

    assert result["success"] is True
    assert result["bytes"] == 100
    assert result["content_type"] == "application/octet-stream"

    assert Path(result["file_path"]).exists()
    assert Path(result["file_path"]).stat().st_size == 100


def test_http_download_404_raises_tool_execution_error(
    tmp_path: Path,
    monkeypatch,
):
    """A 404 response is converted into ToolExecutionError."""

    def fake_get(url, timeout, allow_redirects):
        return FakeResponse(
            content=b"",
            status_code=404,
        )

    monkeypatch.setattr(
        "tools.http_tools.requests.get",
        fake_get,
    )

    save_path = tmp_path / "test_download" / "missing.bin"

    with pytest.raises(ToolExecutionError):
        http_download(
            url="https://example.com/missing.bin",
            save_path=str(save_path),
        )


def test_http_download_creates_parent_directories(
    tmp_path: Path,
    monkeypatch,
):
    """Parent directories are created automatically."""

    def fake_get(url, timeout, allow_redirects):
        return FakeResponse(
            content=b"hello",
            content_type="text/plain",
        )

    monkeypatch.setattr(
        "tools.http_tools.requests.get",
        fake_get,
    )

    save_path = (
        tmp_path
        / "nested"
        / "dir"
        / "file.txt"
    )

    result = http_download(
        url="https://example.com/file.txt",
        save_path=str(save_path),
    )

    assert result["success"] is True
    assert save_path.exists()
    assert save_path.read_bytes() == b"hello"


def test_http_download_custom_timeout(
    tmp_path: Path,
    monkeypatch,
):
    """The configured timeout is passed to requests."""

    observed = {}

    def fake_get(url, timeout, allow_redirects):
        observed["timeout"] = timeout

        return FakeResponse(
            content=b"data",
        )

    monkeypatch.setattr(
        "tools.http_tools.requests.get",
        fake_get,
    )

    save_path = tmp_path / "file.bin"

    http_download(
        url="https://example.com/file.bin",
        save_path=str(save_path),
        timeout=15,
    )

    assert observed["timeout"] == 15
