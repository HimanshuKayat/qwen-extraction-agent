"""Build the controlled ToolRegistry used by the agent.

Phase 1 tools are implemented and enabled.

Future tools are intentionally not registered until their implementations
are ready. This keeps the model's available tool list accurate and prevents
the model from attempting unavailable capabilities.
"""

from __future__ import annotations

from tools import file_tools, http_tools, validation_tools
from tools.registry import ToolRegistry, ToolSpec


def build_registry() -> ToolRegistry:
    """Construct and return the controlled tool registry."""

    registry = ToolRegistry()

    # ------------------------------------------------------------------
    # Phase 1: HTTP / file / validation tools
    # ------------------------------------------------------------------

    registry.register(
        ToolSpec(
            name="http_download",
            description=(
                "Download a file from a URL using HTTP GET and save it "
                "as a raw artifact for the specified source. The agent "
                "must provide the source_id and filename. The storage "
                "directory is controlled by the application."
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
                "such as size, extension, MIME type, and spreadsheet "
                "sheet names."
            ),
            category="file",
            function=file_tools.inspect_file,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Local file path.",
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
                "containing shape, columns, data types, and a "
                "small row sample."
            ),
            category="file",
            function=file_tools.read_csv,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Local CSV file path.",
                    },
                    "sample_rows": {
                        "type": "integer",
                        "description": "Number of sample rows.",
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
                "Read an Excel file and return a compact summary "
                "of sheets, shape, columns, data types, and samples."
            ),
            category="file",
            function=file_tools.read_excel,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Local Excel file path.",
                    },
                    "sheet_name": {
                        "type": ["string", "null"],
                        "description": (
                            "Optional sheet name. "
                            "Defaults to the first sheet."
                        ),
                    },
                    "sample_rows": {
                        "type": "integer",
                        "description": "Number of sample rows.",
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
                        "description": "Local PDF file path.",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum pages to inspect.",
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
                "a compact table summary and sample rows."
            ),
            category="file",
            function=file_tools.extract_pdf_table,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Local PDF file path.",
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
                "Check whether required fields are present "
                "in a list of column names."
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
                "Check that a row count falls within "
                "an expected minimum and maximum."
            ),
            category="validation",
            function=validation_tools.validate_row_count,
            argument_schema={
                "type": "object",
                "properties": {
                    "row_count": {
                        "type": "integer",
                        "minimum": 0,
                    },
                    "minimum": {
                        "type": "integer",
                        "default": 1,
                        "minimum": 0,
                    },
                    "maximum": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                    },
                },
                "required": ["row_count"],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    return registry
