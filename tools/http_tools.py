"""HTTP-based extraction tools.

Phase 1 foundation.

The model chooses the HTTP download action, but it does not control
arbitrary filesystem paths.

Storage location is determined deterministically from:

    source_id + filename

through storage.paths.raw_path().
"""

from __future__ import annotations

from typing import Any

import requests

from core.exceptions import ToolExecutionError
from storage.paths import raw_path


DEFAULT_TIMEOUT_SECONDS = 60


def http_download(
    url: str,
    source_id: str,
    filename: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Download a URL into the source's raw storage directory.

    Args:
        url:
            URL to download.

        source_id:
            Identifier of the configured data source.

        filename:
            Filename to use for the raw artifact.

        timeout:
            HTTP request timeout in seconds.

    Returns:
        Structured download result containing the deterministic
        storage path.

    Raises:
        ToolExecutionError:
            If the HTTP request fails or returns a non-success status.
    """

    if not source_id.strip():
        raise ToolExecutionError(
            message="source_id cannot be empty.",
            error_type="InvalidSourceId",
            recoverable=False,
        )

    if not filename.strip():
        raise ToolExecutionError(
            message="filename cannot be empty.",
            error_type="InvalidFilename",
            recoverable=False,
        )

    # Prevent the model from escaping the raw storage directory.
    filename_path = filename.replace("\\", "/")

    if "/" in filename_path or filename_path in {".", ".."}:
        raise ToolExecutionError(
            message=(
                "filename must contain only a filename, "
                "not a directory path."
            ),
            error_type="InvalidFilename",
            recoverable=False,
        )

    try:
        response = requests.get(
            url,
            timeout=timeout,
            allow_redirects=True,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout as exc:
        raise ToolExecutionError(
            message=(
                f"Request to {url} timed out "
                f"after {timeout}s"
            ),
            error_type="Timeout",
            recoverable=True,
        ) from exc

    except requests.exceptions.HTTPError as exc:
        status_code = (
            exc.response.status_code
            if exc.response is not None
            else None
        )

        raise ToolExecutionError(
            message=(
                f"HTTP error downloading {url}: {exc}"
            ),
            error_type="HTTPError",
            recoverable=(
                status_code is not None
                and status_code >= 500
            ),
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise ToolExecutionError(
            message=(
                f"Request error downloading {url}: {exc}"
            ),
            error_type="RequestException",
            recoverable=True,
        ) from exc

    # ---------------------------------------------------------------
    # Deterministic storage location
    # ---------------------------------------------------------------

    path = raw_path(
        source_id,
        filename,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(
        response.content
    )

    return {
        "success": True,
        "url": url,
        "source_id": source_id,
        "filename": filename,
        "file_path": str(path),
        "bytes": len(response.content),
        "content_type": response.headers.get(
            "content-type",
            "unknown",
        ),
    }
