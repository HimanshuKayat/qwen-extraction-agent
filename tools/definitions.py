"""Builds the ToolRegistry used by the agent.

Phase 1 tools are registered with a real ``function`` and
``enabled=True``. Later-phase tools are registered with
``function=None`` and ``enabled=False`` so that execute_action() will
clearly refuse to run them (ToolDisabledError) rather than silently
doing nothing or crashing.
"""

from __future__ import annotations

from tools import file_tools, http_tools, validation_tools
from tools.registry import ToolRegistry, ToolSpec


def build_registry() -> ToolRegistry:
    """Construct and return the fully-populated tool registry."""
    registry = ToolRegistry()

    # ---------------------------------------------------------------
    # Phase 1: HTTP / file-based extraction foundation (IMPLEMENTED)
    # ---------------------------------------------------------------
    registry.register(ToolSpec(
        name="http_download",
        description="Download a file from a URL via HTTP GET and save it locally.",
        category="http",
        function=http_tools.http_download,
        argument_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to download."},
                "save_path": {"type": "string", "description": "Local path to save the file to."},
                "timeout": {"type": "integer", "description": "Timeout in seconds.", "default": 60},
            },
            "required": ["url", "save_path"],
            "additionalProperties": False,
        },
        enabled=True,
    ))

    registry.register(ToolSpec(
        name="inspect_file",
        description="Inspect a local file and return compact metadata (size, extension, MIME type, sheet names).",
        category="file",
        function=file_tools.inspect_file,
        argument_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local path to the file."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        enabled=True,
    ))

    registry.register(ToolSpec(
        name="read_csv",
        description="Read a CSV file and return a compact summary: shape, columns, dtypes, and a small row sample.",
        category="file",
        function=file_tools.read_csv,
        argument_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local path to the CSV file."},
                "sample_rows": {"type": "integer", "description": "Number of sample rows to include.", "default": 5},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        enabled=True,
    ))

    registry.register(ToolSpec(
        name="read_excel",
        description="Read an Excel file (optionally a specific sheet) and return a compact summary.",
        category="file",
        function=file_tools.read_excel,
        argument_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local path to the Excel file."},
                "sheet_name": {"type": ["string", "null"], "description": "Sheet to read; defaults to the first sheet."},
                "sample_rows": {"type": "integer", "description": "Number of sample rows to include.", "default": 5},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        enabled=True,
    ))

    registry.register(ToolSpec(
        name="read_pdf",
        description="Extract text from the first N pages of a PDF and return page count plus text samples.",
        category="file",
        function=file_tools.read_pdf,
        argument_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local path to the PDF file."},
                "max_pages": {"type": "integer", "description": "Maximum number of pages to preview.", "default": 3},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        enabled=True,
    ))

    registry.register(ToolSpec(
        name="extract_pdf_table",
        description="Extract tables from a single PDF page and return a compact summary with a row sample.",
        category="file",
        function=file_tools.extract_pdf_table,
        argument_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local path to the PDF file."},
                "page_number": {"type": "integer", "description": "1-indexed page number.", "default": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        enabled=True,
    ))

    registry.register(ToolSpec(
        name="validate_required_fields",
        description="Check that required fields are present among a list of column names.",
        category="validation",
        function=validation_tools.validate_required_fields,
        argument_schema={
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}},
                "required_fields": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["columns", "required_fields"],
            "additionalProperties": False,
        },
        enabled=True,
    ))

    registry.register(ToolSpec(
        name="validate_row_count",
        description="Check that a row count falls within an expected minimum/maximum range.",
        category="validation",
        function=validation_tools.validate_row_count,
        argument_schema={
            "type": "object",
            "properties": {
                "row_count": {"type": "integer"},
                "minimum": {"type": "integer", "default": 1},
                "maximum": {"type": ["integer", "null"]},
            },
            "required": ["row_count"],
            "additionalProperties": False,
        },
        enabled=True,
    ))

    # ---------------------------------------------------------------
    # Phase 2-4: registered as PLACEHOLDERS ONLY. function=None means
    # execute_action() will raise ToolDisabledError rather than run
    # anything. Listed here so the model's tool listing accurately
    # reflects what is planned but not yet available.
    # ---------------------------------------------------------------
    _placeholder_specs = [
        ("browser_open", "browser", "Open a URL in a controlled browser session. (Phase 2, not implemented.)"),
        ("browser_inspect", "browser", "Inspect the current page's DOM/content. (Phase 2, not implemented.)"),
        ("browser_click", "browser", "Click an element on the current page. (Phase 2, not implemented.)"),
        ("browser_fill", "browser", "Fill a form field on the current page. (Phase 2, not implemented.)"),
        ("browser_select", "browser", "Select an option in a dropdown. (Phase 2, not implemented.)"),
        ("browser_wait", "browser", "Wait for a condition on the current page. (Phase 2, not implemented.)"),
        ("browser_download", "browser", "Trigger and capture a browser-initiated download. (Phase 2, not implemented.)"),
        ("browser_back", "browser", "Navigate back in browser history. (Phase 2, not implemented.)"),
        ("sparql_query", "query", "Run a SPARQL query against an endpoint such as Wikidata. (Phase 3, not implemented.)"),
        ("api_get", "query", "Call a generic REST API endpoint. (Phase 3, not implemented.)"),
        ("email_search", "email", "Search a mailbox for matching messages. (Phase 4, not implemented.)"),
        ("email_read", "email", "Read the contents of an email message. (Phase 4, not implemented.)"),
        ("email_get_attachment", "email", "Download an attachment from an email message. (Phase 4, not implemented.)"),
        ("email_download_link", "email", "Follow and download a link found inside an email. (Phase 4, not implemented.)"),
    ]
    for name, category, description in _placeholder_specs:
        registry.register(ToolSpec(
            name=name,
            description=description,
            category=category,
            function=None,
            argument_schema={"type": "object", "properties": {}, "additionalProperties": True},
            enabled=False,
        ))

    return registry
