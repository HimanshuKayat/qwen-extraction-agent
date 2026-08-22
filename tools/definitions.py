"""
Definitions for the controlled tool registry.

This module is responsible for constructing the set of tools that the
agent is currently allowed to use.

Architecture:

    agent
      |
      v
    ToolRegistry
      |
      v
    deterministic tools

Only implemented tools are registered here.

Future capabilities such as Playwright, SPARQL, API access, email,
database operations, hashing, monitoring, and scheduling will be added
when their implementations are ready.

The model never receives tools that are not currently implemented.
"""

from __future__ import annotations

from tools import file_tools
from tools import http_tools
from tools import validation_tools

from tools.registry import ToolRegistry, ToolSpec


def build_registry() -> ToolRegistry:
    """
    Build and return the Phase-1 tool registry.

    The registry contains only deterministic tools that are currently
    implemented and safe for the agent to execute.

    Returns:
        ToolRegistry: Registry containing all currently enabled tools.
    """

    registry = ToolRegistry()

    # =============================================================
    # PHASE 1
    # HTTP / FILE EXTRACTION
    # =============================================================

    registry.register(
        ToolSpec(
            name="http_download",
            description=(
                "Download a file from a URL using HTTP GET and "
                "save it to the configured local storage location."
            ),
            category="http",
            function=http_tools.http_download,
            argument_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the file to download.",
                    },
                    "save_path": {
                        "type": "string",
                        "description": (
                            "Path relative to the configured storage "
                            "root where the downloaded file should be saved."
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
                    "save_path",
                ],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    # =============================================================
    # FILE INSPECTION
    # =============================================================

    registry.register(
        ToolSpec(
            name="inspect_file",
            description=(
                "Inspect a downloaded file and return compact metadata "
                "such as filename, extension, size, MIME type, and "
                "available Excel sheets."
            ),
            category="file",
            function=file_tools.inspect_file,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path relative to the configured storage root."
                        ),
                    },
                },
                "required": [
                    "path",
                ],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    # =============================================================
    # CSV
    # =============================================================

    registry.register(
        ToolSpec(
            name="read_csv",
            description=(
                "Read a CSV file and return a compact structural summary "
                "including row count, column names, data types, and a "
                "small sample of rows."
            ),
            category="file",
            function=file_tools.read_csv,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path relative to the configured storage root."
                        ),
                    },
                    "sample_rows": {
                        "type": "integer",
                        "description": (
                            "Number of rows to return in the sample."
                        ),
                        "default": 5,
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": [
                    "path",
                ],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    # =============================================================
    # EXCEL
    # =============================================================

    registry.register(
        ToolSpec(
            name="read_excel",
            description=(
                "Read an Excel workbook and return a compact structural "
                "summary including sheets, columns, data types, and a "
                "small sample of rows."
            ),
            category="file",
            function=file_tools.read_excel,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path relative to the configured storage root."
                        ),
                    },
                    "sheet_name": {
                        "type": [
                            "string",
                            "null",
                        ],
                        "description": (
                            "Excel sheet to inspect. If omitted, "
                            "the first sheet is used."
                        ),
                    },
                    "sample_rows": {
                        "type": "integer",
                        "description": (
                            "Number of rows to return in the sample."
                        ),
                        "default": 5,
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": [
                    "path",
                ],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    # =============================================================
    # PDF TEXT
    # =============================================================

    registry.register(
        ToolSpec(
            name="read_pdf",
            description=(
                "Read text from a PDF and return compact information "
                "about its pages and extracted text."
            ),
            category="file",
            function=file_tools.read_pdf,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path relative to the configured storage root."
                        ),
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": (
                            "Maximum number of PDF pages to inspect."
                        ),
                        "default": 3,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": [
                    "path",
                ],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    # =============================================================
    # PDF TABLE EXTRACTION
    # =============================================================

    registry.register(
        ToolSpec(
            name="extract_pdf_table",
            description=(
                "Extract tables from a PDF page and return a compact "
                "representation of the detected table data."
            ),
            category="file",
            function=file_tools.extract_pdf_table,
            argument_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path relative to the configured storage root."
                        ),
                    },
                    "page_number": {
                        "type": "integer",
                        "description": (
                            "One-based PDF page number."
                        ),
                        "default": 1,
                        "minimum": 1,
                    },
                },
                "required": [
                    "path",
                ],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    # =============================================================
    # VALIDATION
    # =============================================================

    registry.register(
        ToolSpec(
            name="validate_required_fields",
            description=(
                "Check whether all required fields exist in a dataset's "
                "column list."
            ),
            category="validation",
            function=validation_tools.validate_required_fields,
            argument_schema={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "required_fields": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
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

    # =============================================================
    # ROW COUNT VALIDATION
    # =============================================================

    registry.register(
        ToolSpec(
            name="validate_row_count",
            description=(
                "Validate that a dataset row count falls within an "
                "expected minimum and optional maximum."
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
                        "minimum": 0,
                        "default": 1,
                    },
                    "maximum": {
                        "type": [
                            "integer",
                            "null",
                        ],
                        "minimum": 0,
                    },
                },
                "required": [
                    "row_count",
                ],
                "additionalProperties": False,
            },
            enabled=True,
        )
    )

    return registry
