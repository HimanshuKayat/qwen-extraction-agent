"""Unit tests for controlled browser tools."""

import pytest

from core.exceptions import ToolExecutionError
from tools.browser_tools import _validate_url


def test_validate_raw_http_url():
    """A normal HTTP URL is accepted."""

    result = _validate_url(
        "https://example.com/"
    )

    assert result == "https://example.com/"


def test_validate_http_url():
    """HTTP URLs are accepted."""

    result = _validate_url(
        "http://example.com/"
    )

    assert result == "http://example.com/"


def test_reject_markdown_url():
    """Markdown-formatted URLs must never reach Playwright."""

    with pytest.raises(ToolExecutionError) as exc:
        _validate_url(
            "[https://example.com/](https://example.com/)"
        )

    assert exc.value.error_type == "InvalidBrowserURL"


def test_reject_angle_bracket_url():
    """Angle-bracket formatted URLs are rejected."""

    with pytest.raises(ToolExecutionError):
        _validate_url(
            "<https://example.com/>"
        )


def test_reject_empty_url():
    """Empty URLs are rejected."""

    with pytest.raises(ToolExecutionError):
        _validate_url("")


def test_reject_unsupported_scheme():
    """Only HTTP and HTTPS are supported."""

    with pytest.raises(ToolExecutionError):
        _validate_url(
            "ftp://example.com/file"
        )


def test_reject_missing_hostname():
    """URLs without a hostname are rejected."""

    with pytest.raises(ToolExecutionError):
        _validate_url(
            "https://"
        )
