"""
Controlled browser tools.

Phase 2 foundation.

Uses Playwright's asynchronous API so the browser layer works correctly
inside Jupyter/Colab as well as in an asynchronous server environment.

The model chooses browser actions. Playwright performs the actual browser
operation.
""""""
Controlled browser tools.

Phase 2 foundation.

The public tool functions are synchronous so they can be called safely
by the synchronous agent/tool-registry layer.

Internally, Playwright runs on one persistent asyncio event loop in a
dedicated background thread. This is important because a Playwright
Browser/Page must remain associated with the same event loop for the
entire browser session.

Architecture:

    Agent
      |
      v
    execute_action()
      |
      v
    browser_open()
      |
      v
    PersistentAsyncRunner
      |
      v
    Playwright asyncio loop
      |
      v
    Chromium
"""

from __future__ import annotations

import asyncio
import inspect
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
MAX_INSPECT_TEXT = 20_000
MAX_INSPECT_LINKS = 200

T = TypeVar("T")


def _validate_url(url: str) -> str:
    """
    Validate a browser URL.

    Only raw HTTP/HTTPS URLs are accepted.
    Markdown-formatted URLs are explicitly rejected.
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

    # Reject angle-bracket URLs.
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


class AsyncRunner:
    """
    Own one persistent asyncio event loop in a background thread.

    Every Playwright operation is submitted to this same loop.

    This prevents browser/page objects from being used across different
    asyncio event loops.
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
        """Start the persistent event loop if necessary."""

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
        """
        Execute a coroutine on the persistent browser event loop.
        """

        self.start()

        if self._loop is None:
            raise RuntimeError(
                "Browser asyncio loop was not initialized."
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
        """Stop the persistent event loop."""

        with self._lock:
            loop = self._loop
            thread = self._thread

            self._loop = None
            self._thread = None

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(
                loop.stop
            )

        if thread is not None and thread.is_alive():
            thread.join(timeout=10)


class BrowserSession:
    """Owns one persistent Playwright browser session."""

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
            # Close an existing browser first.
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

            body = self._page.locator("body")

            text = await body.inner_text(
                timeout=10_000,
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

    async def close_async(self) -> None:
        """Close the browser session."""

        try:
            if self._browser is not None:
                await self._browser.close()

        except Exception:
            pass

        finally:
            self._browser = None
            self._page = None

        try:
            if self._playwright is not None:
                await self._playwright.stop()

        except Exception:
            pass

        finally:
            self._playwright = None


_BROWSER_RUNNER = AsyncRunner()
_SESSION = BrowserSession()


def browser_open(
    url: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Open a website in the controlled browser.

    Synchronous wrapper around the persistent Playwright event loop.
    """

    return _BROWSER_RUNNER.run(
        _SESSION.open_async(
            url=url,
            timeout=timeout,
        ),
        timeout=timeout + 10,
    )


def browser_inspect() -> dict[str, Any]:
    """
    Inspect the currently open browser page.

    Synchronous wrapper around the persistent Playwright event loop.
    """

    return _BROWSER_RUNNER.run(
        _SESSION.inspect_async(),
        timeout=20,
    )


def browser_close() -> dict[str, Any]:
    """
    Close the current browser session.
    """

    try:
        _BROWSER_RUNNER.run(
            _SESSION.close_async(),
            timeout=15,
        )

        return {
            "success": True,
            "message": "Browser session closed.",
        }

    finally:
        _BROWSER_RUNNER.stop()


# Keep this helper available for tests and diagnostics.
def is_async_tool(function: Any) -> bool:
    """Return whether a function is asynchronous."""

    return inspect.iscoroutinefunction(function)

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
