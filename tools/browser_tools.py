"""
Controlled browser tools.

Phase 2 foundation.

Playwright runs on one persistent asyncio event loop in a dedicated
background thread. Public browser tools are synchronous so they can be
called safely by the synchronous agent/tool-registry layer.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar
from urllib.parse import urlparse

from playwright.async_api import (
    Browser,
    Page,
    Playwright,
    async_playwright,
)

from core.exceptions import ToolExecutionError


DEFAULT_TIMEOUT_SECONDS = 30
INSPECT_TIMEOUT_SECONDS = 10
MAX_INSPECT_TEXT = 20_000
MAX_INSPECT_LINKS = 200

T = TypeVar("T")


# ---------------------------------------------------------------------------
# URL VALIDATION
# ---------------------------------------------------------------------------


def _validate_url(url: str) -> str:
    """Validate that a browser URL is a raw HTTP/HTTPS URL."""

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

    # Reject Markdown links.
    if url.startswith("[") or "](" in url:
        raise ToolExecutionError(
            message=(
                "Browser URL must be a raw URL, not Markdown. "
                f"Received: {url}"
            ),
            error_type="InvalidBrowserURL",
            recoverable=False,
        )

    # Reject angle-bracket links.
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


# ---------------------------------------------------------------------------
# PERSISTENT ASYNC RUNNER
# ---------------------------------------------------------------------------


class AsyncRunner:
    """
    Persistent asyncio event loop for Playwright.

    Browser/page objects are created and used on this same loop for the
    entire lifetime of the browser session.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def _thread_target(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self._loop = loop
        self._ready.set()

        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)

            for task in pending:
                task.cancel()

            if pending:
                loop.run_until_complete(
                    asyncio.gather(
                        *pending,
                        return_exceptions=True,
                    )
                )

            loop.close()

    def start(self) -> None:
        """Start the persistent event loop."""

        with self._lock:
            if (
                self._thread is not None
                and self._thread.is_alive()
                and self._loop is not None
                and self._loop.is_running()
            ):
                return

            self._ready.clear()

            self._thread = threading.Thread(
                target=self._thread_target,
                name="browser-async-loop",
                daemon=True,
            )

            self._thread.start()

        if not self._ready.wait(timeout=10):
            raise RuntimeError(
                "Timed out while starting browser asyncio loop."
            )

    def run(
        self,
        coroutine: Coroutine[Any, Any, T],
        timeout: float | None = None,
    ) -> T:
        """Run a coroutine on the persistent browser loop."""

        self.start()

        if self._loop is None:
            raise RuntimeError(
                "Browser asyncio loop is not available."
            )

        future = asyncio.run_coroutine_threadsafe(
            coroutine,
            self._loop,
        )

        try:
            return future.result(timeout=timeout)

        except Exception:
            if not future.done():
                future.cancel()

            raise

    def stop(self) -> None:
        """Stop the persistent browser loop."""

        with self._lock:
            loop = self._loop
            thread = self._thread

            self._loop = None
            self._thread = None

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

        if thread is not None and thread.is_alive():
            thread.join(timeout=5)


# ---------------------------------------------------------------------------
# BROWSER SESSION
# ---------------------------------------------------------------------------


class BrowserSession:
    """Owns the Playwright browser and current page."""

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    async def open_async(
        self,
        url: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Open a URL in Chromium."""

        url = _validate_url(url)

        if timeout <= 0:
            raise ToolExecutionError(
                message="Browser timeout must be greater than zero.",
                error_type="InvalidBrowserTimeout",
                recoverable=False,
            )

        try:
            await self.close_async()

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
            await self.close_async()
            raise

        except Exception as exc:
            await self.close_async()

            raise ToolExecutionError(
                message=f"Browser failed to open {url}: {exc}",
                error_type="BrowserOpenError",
                recoverable=True,
            ) from exc

    async def inspect_async(self) -> dict[str, Any]:
        """
        Inspect the current page.

        Every browser operation has its own timeout. This prevents one
        Playwright operation from blocking the agent indefinitely.
        """

        if self._page is None:
            raise ToolExecutionError(
                message="No browser page is currently open.",
                error_type="BrowserSessionError",
                recoverable=False,
            )

        page = self._page

        # ---------------------------------------------------------------
        # TITLE
        # ---------------------------------------------------------------

        try:
            title = await asyncio.wait_for(
                page.title(),
                timeout=INSPECT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise ToolExecutionError(
                message=f"Failed to read page title: {exc}",
                error_type="BrowserInspectTitleError",
                recoverable=True,
            ) from exc

        # ---------------------------------------------------------------
        # URL
        # ---------------------------------------------------------------

        try:
            url = page.url
        except Exception as exc:
            raise ToolExecutionError(
                message=f"Failed to read page URL: {exc}",
                error_type="BrowserInspectURLError",
                recoverable=True,
            ) from exc

        # ---------------------------------------------------------------
        # PAGE TEXT
        # ---------------------------------------------------------------

        try:
            text = await asyncio.wait_for(
                page.locator("body").inner_text(
                    timeout=INSPECT_TIMEOUT_SECONDS * 1000,
                ),
                timeout=INSPECT_TIMEOUT_SECONDS + 2,
            )

        except Exception as exc:
            raise ToolExecutionError(
                message=f"Failed to extract page text: {exc}",
                error_type="BrowserInspectTextError",
                recoverable=True,
            ) from exc

        # ---------------------------------------------------------------
        # LINKS
        # ---------------------------------------------------------------

        try:
            links = await asyncio.wait_for(
                page.locator("a").evaluate_all(
                    """
                    elements => elements.map(a => ({
                        text: (a.innerText || "").trim(),
                        href: a.href || ""
                    })).filter(x => x.href)
                    """
                ),
                timeout=INSPECT_TIMEOUT_SECONDS + 2,
            )

        except Exception as exc:
            # Links are useful but not essential. If link extraction fails,
            # return the page text rather than failing the entire inspection.
            links = []

        return {
            "success": True,
            "url": url,
            "title": title,
            "text": text[:MAX_INSPECT_TEXT],
            "links": links[:MAX_INSPECT_LINKS],
            "link_count": len(links),
        }

    async def close_async(self) -> None:
        """Close the browser and Playwright cleanly."""

        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass

        self._browser = None
        self._page = None

        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass

        self._playwright = None


# ---------------------------------------------------------------------------
# GLOBAL BROWSER SESSION
# ---------------------------------------------------------------------------


_BROWSER_RUNNER = AsyncRunner()
_SESSION = BrowserSession()


# ---------------------------------------------------------------------------
# PUBLIC TOOLS
# ---------------------------------------------------------------------------


def browser_open(
    url: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Open a website in the controlled browser."""

    return _BROWSER_RUNNER.run(
        _SESSION.open_async(
            url=url,
            timeout=timeout,
        ),
        timeout=timeout + 10,
    )


def browser_inspect() -> dict[str, Any]:
    """Inspect the currently open webpage."""

    return _BROWSER_RUNNER.run(
        _SESSION.inspect_async(),
        timeout=INSPECT_TIMEOUT_SECONDS + 5,
    )


def browser_close() -> dict[str, Any]:
    """Close the current browser session."""

    try:
        _BROWSER_RUNNER.run(
            _SESSION.close_async(),
            timeout=10,
        )

        return {
            "success": True,
            "message": "Browser session closed.",
        }

    finally:
        _BROWSER_RUNNER.stop()
