"""Controlled browser tools.

Phase 2 foundation.

Uses Playwright's asynchronous API so the browser layer works correctly
inside Jupyter/Colab as well as in an asynchronous server environment.

The model chooses browser actions. Playwright performs the actual browser
operation.
"""

from __future__ import annotations

from typing import Any

from playwright.async_api import (
    Browser,
    Page,
    Playwright,
    async_playwright,
)

from core.exceptions import ToolExecutionError


class BrowserSession:
    """Owns one controlled Playwright browser session."""

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    async def open(
        self,
        url: str,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Open a URL in a controlled Chromium browser."""

        try:
            self._playwright = await async_playwright().start()

            self._browser = await self._playwright.chromium.launch(
                headless=True,
            )

            self._page = await self._browser.new_page()

            response = await self._page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=timeout * 1000,
            )

            return {
                "success": True,
                "url": self._page.url,
                "title": await self._page.title(),
                "status_code": (
                    response.status
                    if response is not None
                    else None
                ),
            }

        except Exception as exc:
            await self.close()

            raise ToolExecutionError(
                message=f"Browser failed to open {url}: {exc}",
                error_type="BrowserOpenError",
                recoverable=True,
            ) from exc

    async def inspect(self) -> dict[str, Any]:
        """Inspect the currently open page."""

        if self._page is None:
            raise ToolExecutionError(
                message="No browser page is currently open.",
                error_type="BrowserSessionError",
                recoverable=False,
            )

        try:
            title = await self._page.title()
            url = self._page.url

            text = await self._page.locator(
                "body"
            ).inner_text(
                timeout=10_000
            )

            links = await self._page.locator(
                "a"
            ).evaluate_all(
                """
                elements => elements.map(a => ({
                    text: (a.innerText || "").trim(),
                    href: a.href
                })).filter(x => x.href)
                """
            )

            return {
                "success": True,
                "url": url,
                "title": title,
                "text": text[:20_000],
                "links": links[:200],
                "link_count": len(links),
            }

        except Exception as exc:
            raise ToolExecutionError(
                message=f"Browser inspection failed: {exc}",
                error_type="BrowserInspectError",
                recoverable=True,
            ) from exc

    async def close(self) -> None:
        """Close the browser session."""

        try:
            if self._browser is not None:
                await self._browser.close()
        finally:
            self._browser = None
            self._page = None

            if self._playwright is not None:
                await self._playwright.stop()

            self._playwright = None


_SESSION = BrowserSession()


async def browser_open(
    url: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """Open a website in the controlled browser."""

    return await _SESSION.open(
        url=url,
        timeout=timeout,
    )


async def browser_inspect() -> dict[str, Any]:
    """Inspect the currently open page."""

    return await _SESSION.inspect()


async def browser_close() -> dict[str, Any]:
    """Close the current browser session."""

    await _SESSION.close()

    return {
        "success": True,
        "message": "Browser session closed.",
    }
