"""Controlled browser tools.

Phase 2.

Uses Playwright's asynchronous API and supports dynamically rendered
websites by waiting for the page to become usable before inspection.
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
INSPECT_WAIT_MS = 2000
MAX_TEXT_LENGTH = 20_000
MAX_LINKS = 200


def _validate_url(url: str) -> str:
    """Validate and normalize a browser URL."""

    if not isinstance(url, str):
        raise ToolExecutionError(
            message="Browser URL must be a string.",
            error_type="InvalidURL",
            recoverable=False,
        )

    url = url.strip()

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ToolExecutionError(
            message=f"Unsupported URL scheme: {parsed.scheme!r}",
            error_type="InvalidURL",
            recoverable=False,
        )

    if not parsed.netloc:
        raise ToolExecutionError(
            message=f"Invalid browser URL: {url}",
            error_type="InvalidURL",
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

        try:
            # Close an existing session before opening another page.
            await self.close()

            self._playwright = await async_playwright().start()

            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            self._page = await self._browser.new_page()

            response = await self._page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=timeout * 1000,
            )

            # Give JavaScript a chance to render the page.
            try:
                await self._page.wait_for_load_state(
                    "networkidle",
                    timeout=10_000,
                )
            except Exception:
                # Some sites never reach networkidle because of analytics,
                # polling, ads, etc. That should not make the navigation fail.
                pass

            await self._page.wait_for_timeout(INSPECT_WAIT_MS)

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
        """Inspect the currently open page after rendering."""

        if self._page is None:
            raise ToolExecutionError(
                message="No browser page is currently open.",
                error_type="BrowserSessionError",
                recoverable=False,
            )

        try:
            # Allow dynamically generated content to settle.
            await self._page.wait_for_timeout(INSPECT_WAIT_MS)

            title = await self._page.title()
            url = self._page.url

            # Wait briefly for a body to exist.
            try:
                await self._page.locator("body").wait_for(
                    state="attached",
                    timeout=10_000,
                )
            except Exception:
                pass

            body = self._page.locator("body")

            text = ""

            try:
                text = await body.inner_text(
                    timeout=10_000,
                )
            except Exception:
                pass

            links = []

            try:
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
            except Exception:
                pass

            # If normal inner_text is empty, try visible textContent.
            if not text.strip():
                try:
                    text = await body.text_content(
                        timeout=5_000,
                    ) or ""
                except Exception:
                    pass

            return {
                "success": True,
                "url": url,
                "title": title,
                "text": text[:MAX_TEXT_LENGTH],
                "links": links[:MAX_LINKS],
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
