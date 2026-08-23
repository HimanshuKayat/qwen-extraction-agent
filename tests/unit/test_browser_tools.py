"""Unit tests for controlled browser tools."""

from __future__ import annotations

import pytest

from tools.browser_tools import (
    _detect_block_type,
    _validate_url,
)


def test_validate_url_accepts_https():
    assert (
        _validate_url("https://example.com/")
        == "https://example.com/"
    )


def test_validate_url_accepts_http():
    assert (
        _validate_url("http://example.com/")
        == "http://example.com/"
    )


def test_validate_url_strips_whitespace():
    assert (
        _validate_url("  https://example.com/  ")
        == "https://example.com/"
    )


def test_validate_url_strips_markdown_link():
    result = _validate_url(
        "[https://example.com/](https://example.com/)"
    )

    assert result == "https://example.com/"


def test_validate_url_rejects_invalid_scheme():
    with pytest.raises(Exception):
        _validate_url("ftp://example.com/file")


def test_detects_npci_tspd_challenge():
    html = """
    <html>
        <script>
            window["failureConfig"] = "TSPD";
            var challenge = "/TSPD/";
        </script>
    </html>
    """

    result = _detect_block_type(
        html=html,
        text="",
        title="",
        url="https://www.npci.org.in/product/upi/product-statistics",
    )

    assert result == "anti_bot_challenge"


def test_detects_captcha():
    result = _detect_block_type(
        html="<html><body>CAPTCHA verification required</body></html>",
        text="CAPTCHA verification required",
        title="Security Check",
        url="https://example.com/",
    )

    assert result == "anti_bot_challenge"


def test_normal_page_is_not_blocked():
    result = _detect_block_type(
        html="""
        <html>
            <body>
                <h1>Example Domain</h1>
                <a href="https://example.com/test">Test</a>
            </body>
        </html>
        """,
        text="Example Domain Test",
        title="Example Domain",
        url="https://example.com/",
    )

    assert result is None
