"""Tests for controlled browser tools."""

import pytest

from tools.browser_tools import (
    browser_close,
    browser_inspect,
    browser_open,
)


@pytest.mark.asyncio
async def test_browser_open():
    result = await browser_open(
        "https://example.com"
    )

    assert result["success"] is True
    assert result["title"] == "Example Domain"
    assert "example.com" in result["url"]


@pytest.mark.asyncio
async def test_browser_inspect():
    await browser_open(
        "https://example.com"
    )

    result = await browser_inspect()

    assert result["success"] is True
    assert result["title"] == "Example Domain"
    assert result["link_count"] >= 1
    assert isinstance(result["links"], list)

    await browser_close()


@pytest.mark.asyncio
async def test_browser_close():
    result = await browser_close()

    assert result["success"] is True
