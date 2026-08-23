"""Builds the ToolRegistry used by the agent.

Phase 1:
    HTTP / file extraction and validation tools.

Phase 2:
    Controlled browser tools using Playwright.

Later phases:
    Query, email, and other integrations remain disabled placeholders.

The model can only execute tools that are explicitly registered and
enabled here.
"""

from __future__ import annotations

from tools import (
    browser_tools,
    file_tools,
    http_tools,
    validation_tools,
)

from tools.registry import ToolRegistry, ToolSpec


def build_registry() -> ToolRegistry:
    """Construct and return the fully populated tool registry."""

    registry = ToolRegistry()

    # ===============================================================
    # PHASE 1 — HTTP / FILE EXTRACTION
    # ===============================================================

    registry.register(
        ToolSpec(
            name="http_download",
            description=(
                "Download a file from a URL using HTTP GET and save "
                "it as a raw artifact."
            ),
            category="http",
            function=http_tools.http_download,
            argument_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to download.",
                    },
                    "source_id": {
                        "type": "string",
                        "description": (
                            "Configured source identifier. "
                            "Used to determine the raw storage directory."
                        ),
                    },
                    "filename": {
                        "type": "string",
                        "description": (
                            "Filename for the downloaded raw artifact. "
                            "Must be a filename only, not a directory path."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "HTTP timeout in seconds.",
                        "default": 60,
                        "minimum": 1,
                    },
                },
                "required": [
                    "url",
                    "source_id",
                    "filename",
                ],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    registry.register(
        ToolSpec(
            name="inspect_file",
            description=(
                "Inspect a local file and return compact metadata "
                "including size, extension, MIME type, and workbook "
                "sheet names where applicable."
            ),
            category="file",
            function=file_tools.inspect_file,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Local path to the file.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    registry.register(
        ToolSpec(
            name="read_csv",
            description=(
                "Read a CSV file and return a compact summary "
                "containing shape, columns, dtypes, and a small "
                "sample of rows."
            ),
            category="file",
            function=file_tools.read_csv,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Local path to the CSV file.",
                    },
                    "sample_rows": {
                        "type": "integer",
                        "description": (
                            "Number of sample rows to include."
                        ),
                        "default": 5,
                        "minimum": 1,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    registry.register(
        ToolSpec(
            name="read_excel",
            description=(
                "Read an Excel file, optionally selecting a sheet, "
                "and return a compact structural summary."
            ),
            category="file",
            function=file_tools.read_excel,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Local path to the Excel file.",
                    },
                    "sheet_name": {
                        "type": ["string", "null"],
                        "description": (
                            "Sheet to read. Defaults to the first sheet."
                        ),
                    },
                    "sample_rows": {
                        "type": "integer",
                        "description": (
                            "Number of sample rows to include."
                        ),
                        "default": 5,
                        "minimum": 1,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    registry.register(
        ToolSpec(
            name="read_pdf",
            description=(
                "Extract text from the first N pages of a PDF "
                "and return page count plus compact text samples."
            ),
            category="file",
            function=file_tools.read_pdf,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Local path to the PDF file.",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": (
                            "Maximum number of pages to preview."
                        ),
                        "default": 3,
                        "minimum": 1,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    registry.register(
        ToolSpec(
            name="extract_pdf_table",
            description=(
                "Extract tables from a single PDF page and return "
                "a compact table summary with sample rows."
            ),
            category="file",
            function=file_tools.extract_pdf_table,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Local path to the PDF file.",
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "1-indexed PDF page number.",
                        "default": 1,
                        "minimum": 1,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    registry.register(
        ToolSpec(
            name="validate_required_fields",
            description=(
                "Check that required fields are present among "
                "a list of column names."
            ),
            category="validation",
            function=validation_tools.validate_required_fields,
            argument_schema={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "required_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "columns",
                    "required_fields",
                ],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    registry.register(
        ToolSpec(
            name="validate_row_count",
            description=(
                "Check that a row count falls within an expected "
                "minimum and optional maximum range."
            ),
            category="validation",
            function=validation_tools.validate_row_count,
            argument_schema={
                "type": "object",
                "properties": {
                    "row_count": {
                        "type": "integer",
                    },
                    "minimum": {
                        "type": "integer",
                        "default": 1,
                    },
                    "maximum": {
                        "type": ["integer", "null"],
                    },
                },
                "required": ["row_count"],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    # ===============================================================
    # PHASE 2 — CONTROLLED BROWSER
    # ===============================================================

    registry.register(
        ToolSpec(
            name="browser_open",
            description=(
                "Open a website in a controlled Chromium browser. "
                "Use this when the source is a webpage rather than "
                "a direct downloadable file."
            ),
            category="browser",
            function=browser_tools.browser_open,
            argument_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Website URL to open."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Maximum page-load timeout in seconds."
                        ),
                        "default": 30,
                        "minimum": 1,
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    registry.register(
        ToolSpec(
            name="browser_inspect",
            description=(
                "Inspect the currently open webpage and return "
                "its title, URL, visible text, and discovered links. "
                "Use this to identify downloadable files or relevant "
                "navigation targets."
            ),
            category="browser",
            function=browser_tools.browser_inspect,
            argument_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    registry.register(
        ToolSpec(
            name="browser_close",
            description=(
                "Close the current controlled browser session "
                "after browser work is complete."
            ),
            category="browser",
            function=browser_tools.browser_close,
            argument_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    # ===============================================================
    # FUTURE PHASES — DISABLED PLACEHOLDERS
    # ===============================================================

    placeholder_specs = [
        (
            "browser_click",
            "browser",
            "Click an element on the current page. "
            "(Future Phase 2 feature.)",
        ),
        (
            "browser_fill",
            "browser",
            "Fill a form field on the current page. "
            "(Future Phase 2 feature.)",
        ),
        (
            "browser_select",
            "browser",
            "Select an option in a dropdown. "
            "(Future Phase 2 feature.)",
        ),
        (
            "browser_wait",
            "browser",
            "Wait for a condition on the current page. "
            "(Future Phase 2 feature.)",
        ),
        (
            "browser_download",
            "browser",
            "Trigger and capture a browser-initiated download. "
            "(Future Phase 2 feature.)",
        ),
        (
            "browser_back",
            "browser",
            "Navigate back in browser history. "
            "(Future Phase 2 feature.)",
        ),
        (
            "sparql_query",
            "query",
            "Run a SPARQL query against an endpoint such as "
            "Wikidata. (Future Phase 3 feature.)",
        ),
        (
            "api_get",
            "query",
            "Call a controlled REST API endpoint. "
            "(Future Phase 3 feature.)",
        ),
        (
            "email_search",
            "email",
            "Search a mailbox for matching messages. "
            "(Future Phase 4 feature.)",
        ),
        (
            "email_read",
            "email",
            "Read the contents of an email message. "
            "(Future Phase 4 feature.)",
        ),
        (
            "email_get_attachment",
            "email",
            "Download an attachment from an email message. "
            "(Future Phase 4 feature.)",
        ),
        (
            "email_download_link",
            "email",
            "Follow and download a link found inside an email. "
            "(Future Phase 4 feature.)",
        ),
    ]

    for name, category, description in placeholder_specs:
        registry.register(
            ToolSpec(
                name=name,
                description=description,
                category=category,
                function=None,
                argument_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
                enabled=False,
            )
        )

    return registry
