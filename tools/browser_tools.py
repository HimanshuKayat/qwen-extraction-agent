"""Controlled browser tools.

Phase 2 foundation.

These tools provide deterministic browser interaction through Playwright.
The LLM decides WHICH browser action to perform; Playwright performs the
actual browser operation.

No arbitrary Python, shell commands, or browser JavaScript execution is
exposed to the model.
"""

from __future__ import annotations

from typing import Any, Dict

from playwright.sync_api import (
    Browser,
    Page,
    Playwright,
    sync_playwright,
)

from core.exceptions import ToolExecutionError


class BrowserSession:
    """Owns a single controlled Playwright browser session."""

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    def open(self, url: str, timeout: int = 30) -> Dict[str, Any]:
        """Open a URL in a controlled Chromium browser."""

        try:
            self._playwright = sync_playwright().start()

            self._browser = self._playwright.chromium.launch(
                headless=True,
            )

            self._page = self._browser.new_page()

            response = self._page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=timeout * 1000,
            )

            return {
                "success": True,
                "url": self._page.url,
                "title": self._page.title(),
                "status_code": (
                    response.status
                    if response is not None
                    else None
                ),
            }

        except Exception as exc:
            self.close()

            raise ToolExecutionError(
                message=f"Browser failed to open {url}: {exc}",
                error_type="BrowserOpenError",
                recoverable=True,
            ) from exc

    def inspect(self) -> Dict[str, Any]:
        """Return compact information about the current page."""

        if self._page is None:
            raise ToolExecutionError(
                message="No browser page is currently open.",
                error_type="BrowserSessionError",
                recoverable=False,
            )

        try:
            title = self._page.title()
            url = self._page.url

            text = self._page.locator("body").inner_text(
                timeout=10_000
            )

            links = self._page.locator("a").evaluate_all(
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

    def close(self) -> None:
        """Close the browser session."""

        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            self._browser = None
            self._page = None

            if self._playwright is not None:
                self._playwright.stop()

            self._playwright = None


_SESSION = BrowserSession()


def browser_open(
    url: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Open a website in the controlled browser."""

    return _SESSION.open(
        url=url,
        timeout=timeout,
    )


def browser_inspect() -> Dict[str, Any]:
    """Inspect the currently open page."""

    return _SESSION.inspect()


def browser_close() -> Dict[str, Any]:
    """Close the current browser session."""

    _SESSION.close()

    return {
        "success": True,
        "message": "Browser session closed.",
    }
