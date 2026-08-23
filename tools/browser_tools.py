"""
Controlled browser tools.

Phase 2 foundation.

Uses Playwright's asynchronous API so the browser layer works correctly
inside Jupyter/Colab as well as in an asynchronous server environment.

The model chooses browser actions. Playwright performs the actual browser
operation.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from playwright.async_api import (
    Browser,
    Page,
    Playwright,
    async_playwright,
)

from core.exceptions import ToolExecutionError


DEFAULT_TIMEOUT_SECONDS = 30
MAX_INSPECT_TEXT = 20_000
MAX_INSPECT_LINKS = 200


def _validate_url(url: str) -> str:
    """
    Validate and normalize a browser URL.

    Browser tools must receive a real URL, not Markdown such as:

        [https://example.com/](https://example.com/)

    Returns:
        The validated URL.

    Raises:
        ToolExecutionError: If the URL is malformed or unsupported.
    """

    if not isinstance(url, str):
        raise ToolExecutionError(
            message="Browser URL must be a string.",
            error_type="InvalidBrowserURL",
            recoverable=False,
        )

    url = url.strip()

    if not url:
        raise ToolExecutionError(
            message="Browser URL cannot be empty.",
            error_type="InvalidBrowserURL",
            recoverable=False,
        )

    # Explicitly reject common Markdown URL formats.
    if url.startswith("[") or "](" in url:
        raise ToolExecutionError(
            message=(
                "Browser URL must be a raw URL, not Markdown. "
                f"Received: {url}"
            ),
            error_type="InvalidBrowserURL",
            recoverable=False,
        )

    if url.startswith("<") and url.endswith(">"):
        raise ToolExecutionError(
            message=(
                "Browser URL must be a raw URL, not an angle-bracket "
                f"formatted URL. Received: {url}"
            ),
            error_type="InvalidBrowserURL",
            recoverable=False,
        )

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ToolExecutionError(
            message=(
                "Browser URL must use http:// or https://. "
                f"Received: {url}"
            ),
            error_type="InvalidBrowserURL",
            recoverable=False,
        )

    if not parsed.netloc:
        raise ToolExecutionError(
            message=f"Browser URL has no valid hostname: {url}",
            error_type="InvalidBrowserURL",
            recoverable=False,
        )

    return url


class BrowserSession:
    """Owns one controlled Playwright browser session."""

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    async def open(
        self,
        url: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Open a URL in a controlled Chromium browser."""

        url = _validate_url(url)

        if timeout <= 0:
            raise ToolExecutionError(
                message="Browser timeout must be greater than zero.",
                error_type="InvalidBrowserTimeout",
                recoverable=False,
            )

        try:
            # Close an existing session before opening a new one.
            await self.close()

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

        except ToolExecutionError:
            await self.close()
            raise

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
                "text": text[:MAX_INSPECT_TEXT],
                "links": links[:MAX_INSPECT_LINKS],
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
        except Exception:
            # Cleanup must never mask the original operation error.
            pass
        finally:
            self._browser = None
            self._page = None

            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass

            self._playwright = None


_SESSION = BrowserSession()


async def browser_open(
    url: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
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
