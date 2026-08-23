"""Controlled browser tools.

Phase 2 browser foundation.

Uses Playwright's asynchronous API internally.

The browser layer:
- opens webpages
- inspects rendered content
- extracts links
- detects anti-bot/challenge pages
- detects empty/unusable pages
- closes browser sessions

It does NOT attempt to bypass or defeat anti-bot protections.
"""

from __future__ import annotations

import re
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

# Keep these deliberately short.
NETWORK_IDLE_TIMEOUT_MS = 5_000
RENDER_WAIT_MS = 1_000
INSPECT_WAIT_MS = 500

MAX_TEXT_LENGTH = 20_000
MAX_HTML_PREVIEW_LENGTH = 50_000
MAX_LINKS = 200


# ============================================================================
# URL NORMALIZATION / VALIDATION
# ============================================================================


def _normalize_url(url: str) -> str:
    """Normalize URLs returned by the model."""

    if not isinstance(url, str):
        raise ToolExecutionError(
            message="Browser URL must be a string.",
            error_type="InvalidURL",
            recoverable=False,
        )

    url = url.strip()

    # ---------------------------------------------------------
    # Remove surrounding quotes
    # ---------------------------------------------------------

    if (
        len(url) >= 2
        and url[0] == url[-1]
        and url[0] in {"'", '"'}
    ):
        url = url[1:-1].strip()

    # ---------------------------------------------------------
    # Markdown link:
    #
    # [https://example.com/](https://example.com/)
    #
    # ---------------------------------------------------------

    markdown_match = re.fullmatch(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        url,
        flags=re.IGNORECASE,
    )

    if markdown_match:
        url = markdown_match.group(2).strip()

    # ---------------------------------------------------------
    # Sometimes the model returns:
    #
    # <https://example.com/>
    # ---------------------------------------------------------

    if (
        url.startswith("<")
        and url.endswith(">")
        and url[1:-1].strip().lower().startswith(
            ("http://", "https://")
        )
    ):
        url = url[1:-1].strip()

    # ---------------------------------------------------------
    # Remove accidental whitespace
    # ---------------------------------------------------------

    url = url.strip()

    return url


def _validate_url(url: str) -> str:
    """Validate and normalize a browser URL."""

    url = _normalize_url(url)

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


# ============================================================================
# ANTI-BOT / CHALLENGE DETECTION
# ============================================================================


def _detect_block_type(
    html: str,
    text: str,
    title: str,
    url: str,
) -> str | None:
    """Detect common anti-bot/security challenge pages.

    Detection only.

    This function does not attempt to bypass or solve
    any security mechanism.
    """

    combined = "\n".join(
        [
            html or "",
            text or "",
            title or "",
            url or "",
        ]
    ).lower()

    # ---------------------------------------------------------
    # TSPD / NPCI-style protection
    # ---------------------------------------------------------

    tspd_indicators = (
        "/tspd/",
        "tspd_",
        "failureconfig",
        "challenge.support_id",
    )

    for indicator in tspd_indicators:
        if indicator in combined:
            return "anti_bot_challenge"

    # ---------------------------------------------------------
    # Generic anti-bot indicators
    # ---------------------------------------------------------

    challenge_indicators = (
        "captcha",
        "verify you are human",
        "verification required",
        "checking your browser",
        "checking your browser before accessing",
        "security check",
        "access denied",
        "bot detection",
        "automated access",
        "are you a robot",
        "human verification",
        "cloudflare",
        "perimeterx",
        "imperva",
        "incapsula",
    )

    for indicator in challenge_indicators:
        if indicator in combined:
            return "anti_bot_challenge"

    return None


# ============================================================================
# BROWSER SESSION
# ============================================================================


class BrowserSession:
    """Owns one controlled Playwright browser session."""

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    # ------------------------------------------------------------------
    # OPEN
    # ------------------------------------------------------------------

    async def open(
        self,
        url: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Open a URL in controlled Chromium."""

        url = _validate_url(url)

        try:
            # Always start clean.
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

            # ---------------------------------------------------------
            # Give JavaScript a short opportunity to render.
            #
            # Do NOT wait indefinitely for networkidle.
            # NPCI's challenge page may keep connections alive.
            # ---------------------------------------------------------

            try:
                await self._page.wait_for_load_state(
                    "networkidle",
                    timeout=NETWORK_IDLE_TIMEOUT_MS,
                )
            except Exception:
                pass

            await self._page.wait_for_timeout(
                RENDER_WAIT_MS
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

    # ------------------------------------------------------------------
    # INSPECT
    # ------------------------------------------------------------------

    async def inspect(self) -> dict[str, Any]:
        """Inspect the currently open webpage.

        Inspection is deliberately bounded so an anti-bot page,
        iframe, or JavaScript challenge cannot stall the agent.
        """

        if self._page is None:
            raise ToolExecutionError(
                message="No browser page is currently open.",
                error_type="BrowserSessionError",
                recoverable=False,
            )

        try:
            # ---------------------------------------------------------
            # Very short render wait.
            # ---------------------------------------------------------

            await self._page.wait_for_timeout(
                INSPECT_WAIT_MS
            )

            # ---------------------------------------------------------
            # Basic information
            # ---------------------------------------------------------

            title = ""

            try:
                title = await self._page.title()
            except Exception:
                pass

            url = self._page.url

            # ---------------------------------------------------------
            # Get complete HTML.
            #
            # This is the important part for NPCI because the body
            # can be effectively empty while the challenge code is
            # present in the document.
            # ---------------------------------------------------------

            html = ""

            try:
                html = await self._page.content()
            except Exception:
                pass

            # ---------------------------------------------------------
            # Visible text.
            #
            # IMPORTANT:
            # Do not wait 10 seconds here.
            # A body already exists on a loaded page.
            # ---------------------------------------------------------

            text = ""

            try:
                text = await self._page.locator(
                    "body"
                ).inner_text(
                    timeout=2_000,
                )
            except Exception:
                pass

            # ---------------------------------------------------------
            # Fast fallback to text_content.
            # ---------------------------------------------------------

            if not text.strip():

                try:
                    text = (
                        await self._page.locator(
                            "body"
                        ).text_content(
                            timeout=1_000,
                        )
                        or ""
                    )
                except Exception:
                    pass

            # ---------------------------------------------------------
            # Links
            # ---------------------------------------------------------

            links: list[dict[str, str]] = []

            try:
                links = await self._page.locator(
                    "a"
                ).evaluate_all(
                    """
                    elements => elements
                        .map(a => ({
                            text: (a.innerText || "").trim(),
                            href: a.href
                        }))
                        .filter(x => x.href)
                    """
                )
            except Exception:
                links = []

            # ---------------------------------------------------------
            # Iframes
            # ---------------------------------------------------------

            iframe_count = 0

            try:
                iframe_count = len(
                    self._page.frames
                )
            except Exception:
                iframe_count = 0

            # ---------------------------------------------------------
            # Detect anti-bot challenge BEFORE empty-page detection.
            # ---------------------------------------------------------

            block_type = _detect_block_type(
                html=html,
                text=text,
                title=title,
                url=url,
            )

            if block_type is not None:

                return {
                    "success": False,
                    "blocked": True,
                    "block_type": block_type,
                    "recoverable": True,
                    "url": url,
                    "title": title,
                    "text": text[:MAX_TEXT_LENGTH],
                    "links": links[:MAX_LINKS],
                    "link_count": len(links),
                    "iframe_count": iframe_count,
                    "html_length": len(html),
                    "html_preview": (
                        html[:MAX_HTML_PREVIEW_LENGTH]
                    ),
                    "message": (
                        "The website returned an anti-bot "
                        "or security challenge instead of "
                        "usable page content. The browser "
                        "tool does not attempt to bypass "
                        "the challenge."
                    ),
                }

            # ---------------------------------------------------------
            # Empty/unusable page
            # ---------------------------------------------------------

            if not text.strip() and not links:

                return {
                    "success": False,
                    "blocked": False,
                    "empty_page": True,
                    "recoverable": True,
                    "url": url,
                    "title": title,
                    "text": "",
                    "links": [],
                    "link_count": 0,
                    "iframe_count": iframe_count,
                    "html_length": len(html),
                    "html_preview": (
                        html[:MAX_HTML_PREVIEW_LENGTH]
                    ),
                    "message": (
                        "The browser received the page, "
                        "but no usable text or links "
                        "were available."
                    ),
                }

            # ---------------------------------------------------------
            # Normal successful inspection
            # ---------------------------------------------------------

            return {
                "success": True,
                "blocked": False,
                "empty_page": False,
                "recoverable": True,
                "url": url,
                "title": title,
                "text": text[:MAX_TEXT_LENGTH],
                "links": links[:MAX_LINKS],
                "link_count": len(links),
                "iframe_count": iframe_count,
                "html_length": len(html),
                "html_preview": (
                    html[:MAX_HTML_PREVIEW_LENGTH]
                ),
            }

        except Exception as exc:

            raise ToolExecutionError(
                message=f"Browser inspection failed: {exc}",
                error_type="BrowserInspectError",
                recoverable=True,
            ) from exc

    # ------------------------------------------------------------------
    # CLOSE
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the browser session safely."""

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


# ============================================================================
# SHARED SESSION
# ============================================================================


_SESSION = BrowserSession()


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================


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
