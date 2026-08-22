"""Placeholder declarations for tools belonging to later phases.

NONE of the functions in this module are implemented. They exist only
so that the tool registry can describe (in tools/definitions.py) that
these capabilities are planned but not yet available, and so that the
agent's tool listing is honest about what exists.

Calling any of these raises NotImplementedError immediately. They are
never wired into the registry with a callable ``function`` — see
tools/definitions.py, where these are registered with ``function=None``
and ``enabled=False`` so execute_action() refuses to run them.

Phase 2 - Browser / Playwright:
    browser_open, browser_inspect, browser_click, browser_fill,
    browser_select, browser_wait, browser_download, browser_back

Phase 3 - Query/API:
    sparql_query, api_get

Phase 4 - Email:
    email_search, email_read, email_get_attachment, email_download_link
"""

from __future__ import annotations

from typing import Any, Dict


def _not_implemented(name: str) -> Dict[str, Any]:
    raise NotImplementedError(
        f"'{name}' is a Phase 2+ tool and is not implemented in this repository. "
        "It is listed in the tool registry as disabled/placeholder only."
    )


def browser_open(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 2 placeholder. Not implemented."""
    return _not_implemented("browser_open")


def browser_inspect(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 2 placeholder. Not implemented."""
    return _not_implemented("browser_inspect")


def browser_click(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 2 placeholder. Not implemented."""
    return _not_implemented("browser_click")


def browser_fill(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 2 placeholder. Not implemented."""
    return _not_implemented("browser_fill")


def browser_select(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 2 placeholder. Not implemented."""
    return _not_implemented("browser_select")


def browser_wait(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 2 placeholder. Not implemented."""
    return _not_implemented("browser_wait")


def browser_download(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 2 placeholder. Not implemented."""
    return _not_implemented("browser_download")


def browser_back(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 2 placeholder. Not implemented."""
    return _not_implemented("browser_back")


def sparql_query(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 3 placeholder. Not implemented."""
    return _not_implemented("sparql_query")


def api_get(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 3 placeholder. Not implemented."""
    return _not_implemented("api_get")


def email_search(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 4 placeholder. Not implemented."""
    return _not_implemented("email_search")


def email_read(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 4 placeholder. Not implemented."""
    return _not_implemented("email_read")


def email_get_attachment(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 4 placeholder. Not implemented."""
    return _not_implemented("email_get_attachment")


def email_download_link(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Phase 4 placeholder. Not implemented."""
    return _not_implemented("email_download_link")
