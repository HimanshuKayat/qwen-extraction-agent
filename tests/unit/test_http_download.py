"""Unit tests for tools.http_tools.http_download.

Network access is mocked so the unit tests remain deterministic and do not
depend on external services.

The tests also verify that download paths are controlled by the storage
layer rather than supplied directly by the model.
"""

from pathlib import Path

import pytest

from core.exceptions import ToolExecutionError
from storage.paths import raw_path
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


def test_http_download_success(
    tmp_path: Path,
    monkeypatch,
):
    """A successful HTTP response is saved under the raw storage path."""

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

    # Redirect storage root to the pytest temporary directory.
    monkeypatch.setattr(
        "storage.paths.STORAGE_ROOT",
        tmp_path / "storage",
    )
    monkeypatch.setattr(
        "storage.paths.RAW_ROOT",
        tmp_path / "storage" / "raw",
    )

    source_id = "test_download"
    filename = "sample.bin"

    result = http_download(
        url="https://example.com/test.bin",
        source_id=source_id,
        filename=filename,
    )

    expected_path = raw_path(
        source_id,
        filename,
    )

    assert result["success"] is True
    assert result["bytes"] == 100
    assert result["content_type"] == "application/octet-stream"

    assert Path(result["file_path"]) == expected_path
    assert expected_path.exists()
    assert expected_path.stat().st_size == 100


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

    monkeypatch.setattr(
        "storage.paths.STORAGE_ROOT",
        tmp_path / "storage",
    )
    monkeypatch.setattr(
        "storage.paths.RAW_ROOT",
        tmp_path / "storage" / "raw",
    )

    with pytest.raises(ToolExecutionError):
        http_download(
            url="https://example.com/missing.bin",
            source_id="test_download",
            filename="missing.bin",
        )


def test_http_download_creates_source_directory(
    tmp_path: Path,
    monkeypatch,
):
    """The source-specific raw directory is created automatically."""

    def fake_get(url, timeout, allow_redirects):
        return FakeResponse(
            content=b"hello",
            content_type="text/plain",
        )

    monkeypatch.setattr(
        "tools.http_tools.requests.get",
        fake_get,
    )

    monkeypatch.setattr(
        "storage.paths.STORAGE_ROOT",
        tmp_path / "storage",
    )
    monkeypatch.setattr(
        "storage.paths.RAW_ROOT",
        tmp_path / "storage" / "raw",
    )

    source_id = "nested_test"
    filename = "file.txt"

    expected_path = raw_path(
        source_id,
        filename,
    )

    assert not expected_path.parent.exists()

    result = http_download(
        url="https://example.com/file.txt",
        source_id=source_id,
        filename=filename,
    )

    assert result["success"] is True
    assert expected_path.exists()
    assert expected_path.read_bytes() == b"hello"


def test_http_download_custom_timeout(
    tmp_path: Path,
    monkeypatch,
):
    """The configured timeout is passed to requests."""

    observed = {}

    def fake_get(url, timeout, allow_redirects):
        observed["url"] = url
        observed["timeout"] = timeout
        observed["allow_redirects"] = allow_redirects

        return FakeResponse(
            content=b"data",
        )

    monkeypatch.setattr(
        "tools.http_tools.requests.get",
        fake_get,
    )

    monkeypatch.setattr(
        "storage.paths.STORAGE_ROOT",
        tmp_path / "storage",
    )
    monkeypatch.setattr(
        "storage.paths.RAW_ROOT",
        tmp_path / "storage" / "raw",
    )

    http_download(
        url="https://example.com/file.bin",
        source_id="timeout_test",
        filename="file.bin",
        timeout=15,
    )

    assert observed["url"] == "https://example.com/file.bin"
    assert observed["timeout"] == 15
    assert observed["allow_redirects"] is True
