"""HTTP-based tools. Phase 1 foundation.

Deterministic, simple, no LLM reasoning. The agent selects this tool by
name; this module has no awareness of the model at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import requests

from core.exceptions import ToolExecutionError

DEFAULT_TIMEOUT_SECONDS = 60


def http_download(url: str, save_path: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Download a URL to a local path, following redirects.

    Args:
        url: The URL to download.
        save_path: Where to write the downloaded bytes, relative to the
            project's storage root (parent directories are created).
        timeout: Request timeout in seconds.

    Returns:
        A structured result dict with success, url, file_path, bytes,
        and content_type.

    Raises:
        ToolExecutionError: On network errors or non-2xx responses. The
            registry catches this and turns it into a structured
            observation for the model.
    """
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise ToolExecutionError(
            message=f"Request to {url} timed out after {timeout}s",
            error_type="Timeout",
            recoverable=True,
        ) from exc
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        raise ToolExecutionError(
            message=f"HTTP error downloading {url}: {exc}",
            error_type="HTTPError",
            recoverable=status_code is not None and status_code >= 500,
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ToolExecutionError(
            message=f"Request error downloading {url}: {exc}",
            error_type="RequestException",
            recoverable=True,
        ) from exc

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)

    return {
        "success": True,
        "url": url,
        "file_path": str(path),
        "bytes": len(response.content),
        "content_type": response.headers.get("content-type", "unknown"),
    }
