"""File inspection and reading tools. Phase 1 foundation.

All tools here return COMPACT summaries, never entire datasets. The
model only ever sees metadata, shapes, column names, and small samples,
per the "never dump millions of rows into the model context" principle.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from core.exceptions import ToolExecutionError

MAX_SAMPLE_ROWS = 5


def _require_file(path: str) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise ToolExecutionError(
            message=f"File not found: {path}",
            error_type="FileNotFoundError",
            recoverable=False,
        )
    if not file_path.is_file():
        raise ToolExecutionError(
            message=f"Path is not a file: {path}",
            error_type="NotAFileError",
            recoverable=False,
        )
    return file_path


def inspect_file(path: str) -> Dict[str, Any]:
    """Return compact metadata about a downloaded file.

    Includes size, extension, guessed MIME type, and, for spreadsheet
    files, sheet names. Does not read full file contents into memory
    for large binary files.
    """
    file_path = _require_file(path)
    guessed_type, _ = mimetypes.guess_type(str(file_path))
    info: Dict[str, Any] = {
        "success": True,
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "extension": file_path.suffix.lower(),
        "guessed_mime_type": guessed_type or "unknown",
    }

    if file_path.suffix.lower() in (".xlsx", ".xls"):
        try:
            excel_file = pd.ExcelFile(file_path)
            info["sheet_names"] = excel_file.sheet_names
        except Exception as exc:  # noqa: BLE001
            info["sheet_read_error"] = str(exc)

    return info


def read_csv(path: str, sample_rows: int = MAX_SAMPLE_ROWS) -> Dict[str, Any]:
    """Read a CSV file and return a compact summary: shape, columns,
    dtypes, and a small sample of rows. Never returns the full dataframe.
    """
    file_path = _require_file(path)
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(
            message=f"Failed to read CSV {path}: {exc}",
            error_type="CsvReadError",
            recoverable=False,
        ) from exc

    return _dataframe_summary(df, sample_rows)


def read_excel(path: str, sheet_name: str | None = None, sample_rows: int = MAX_SAMPLE_ROWS) -> Dict[str, Any]:
    """Read an Excel file (optionally a specific sheet) and return a
    compact summary. Never returns the full dataframe.
    """
    file_path = _require_file(path)
    try:
        if sheet_name is None:
            excel_file = pd.ExcelFile(file_path)
            sheet_name = excel_file.sheet_names[0]
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(
            message=f"Failed to read Excel {path}: {exc}",
            error_type="ExcelReadError",
            recoverable=False,
        ) from exc

    summary = _dataframe_summary(df, sample_rows)
    summary["sheet_name"] = sheet_name
    return summary


def read_pdf(path: str, max_pages: int = 3) -> Dict[str, Any]:
    """Extract text from the first ``max_pages`` pages of a PDF and
    return page count plus a short text sample per page.
    """
    file_path = _require_file(path)
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - dependency should be installed
        raise ToolExecutionError(
            message="pdfplumber is not installed",
            error_type="MissingDependency",
            recoverable=False,
        ) from exc

    try:
        pages_preview: List[Dict[str, Any]] = []
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages[:max_pages]:
                text = page.extract_text() or ""
                pages_preview.append({
                    "page_number": page.page_number,
                    "text_sample": text[:500],
                })
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(
            message=f"Failed to read PDF {path}: {exc}",
            error_type="PdfReadError",
            recoverable=False,
        ) from exc

    return {
        "success": True,
        "path": str(file_path),
        "total_pages": total_pages,
        "pages_previewed": len(pages_preview),
        "pages": pages_preview,
    }


def extract_pdf_table(path: str, page_number: int = 1) -> Dict[str, Any]:
    """Extract tables from a single PDF page and return a compact summary
    (row/column counts and a small sample), not the full table data.
    """
    file_path = _require_file(path)
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise ToolExecutionError(
            message="pdfplumber is not installed",
            error_type="MissingDependency",
            recoverable=False,
        ) from exc

    try:
        with pdfplumber.open(file_path) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                raise ToolExecutionError(
                    message=f"page_number {page_number} out of range (1-{len(pdf.pages)})",
                    error_type="InvalidPageNumber",
                    recoverable=False,
                )
            page = pdf.pages[page_number - 1]
            tables = page.extract_tables()
    except ToolExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(
            message=f"Failed to extract tables from {path}: {exc}",
            error_type="PdfTableExtractError",
            recoverable=False,
        ) from exc

    table_summaries = []
    for table in tables:
        row_count = len(table)
        col_count = len(table[0]) if table else 0
        sample = table[:MAX_SAMPLE_ROWS]
        table_summaries.append({
            "row_count": row_count,
            "col_count": col_count,
            "sample_rows": sample,
        })

    return {
        "success": True,
        "path": str(file_path),
        "page_number": page_number,
        "table_count": len(tables),
        "tables": table_summaries,
    }


def _dataframe_summary(df: pd.DataFrame, sample_rows: int) -> Dict[str, Any]:
    return {
        "success": True,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns.astype(str)),
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "sample_rows": df.head(sample_rows).to_dict(orient="records"),
    }
