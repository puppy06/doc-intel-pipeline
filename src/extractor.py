"""
Extraction engine for the Claims Document Intelligence Pipeline.

Uses Google Gemini as a Vision-Language Model to process PDF pages rendered
as images, returning structured data that conforms to a schema provided at
call time (either a predefined schema for known types or a dynamically
discovered schema for novel document types).

Architecture notes (Compass Pillars -- Modern Data Foundation)
--------------------------------------------------------------
SEARCH  : Each extracted document can be chunked and embedded into a vector
          database (e.g. Pinecone, Weaviate, pgvector).  At query time an
          LLM retrieval-augmented generation (RAG) chain would:
            1. Embed the user question.
            2. Retrieve top-k similar document chunks.
            3. Feed chunks + question to the LLM for a grounded answer.
          This enables legal analysts to ask natural-language questions
          across thousands of ingested claims documents.

AUTOMATION : In a production deployment this module becomes a task inside
             an orchestrated ETL/ELT pipeline (e.g. Apache Airflow, Prefect,
             or Dagster).  A typical DAG would be:
               ingest_blob -> render_pages -> discover_schema ->
               extract_with_llm -> validate -> load_to_warehouse
             Retry policies, dead-letter queues, and idempotent task IDs
             ensure reliability at scale.

INSIGHTS : The JSON output (ProcessingResult) maps directly to a star-schema
           fact table (one row per document, foreign keys to dim_client,
           dim_vendor, dim_date).  BI tools such as Power BI, Tableau, or
           Metabase can consume the warehouse tables to power dashboards
           showing claim volumes, average invoice amounts, overdue payments,
           and tax-compliance summaries -- all refreshed automatically by the
           pipeline above.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, BinaryIO

import fitz  # PyMuPDF
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

logger = logging.getLogger("doc_intel.extractor")

_DEFAULT_MODEL = "gemini-2.5-flash"

_RENDER_DPI = 150

_MAX_RETRIES = 3
_BASE_RETRY_DELAY = 10


# ---------------------------------------------------------------------------
# PDF rendering (public so main.py can render once and share with discovery)
# ---------------------------------------------------------------------------

def render_pdf_pages(
    pdf_source: bytes | BinaryIO | str | Path,
    *,
    dpi: int = _RENDER_DPI,
) -> list[bytes]:
    """Convert a PDF into a list of PNG byte strings, one per page."""

    if isinstance(pdf_source, (str, Path)):
        pdf_bytes = Path(pdf_source).read_bytes()
    elif isinstance(pdf_source, bytes):
        pdf_bytes = pdf_source
    else:
        pdf_bytes = pdf_source.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[bytes] = []

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        images.append(pix.tobytes(output="png"))

    doc.close()
    return images


# ---------------------------------------------------------------------------
# Dynamic prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(schema: dict[str, Any]) -> str:
    """Build the extraction system prompt from an arbitrary schema dict."""

    schema_block = json.dumps(schema, indent=2, ensure_ascii=False)

    return f"""\
You are a document-intelligence extraction agent for a legal claims firm.

Given one or more images of a PDF document, extract ALL of the following
fields for EACH distinct document found in the pages.

Return a JSON ARRAY of objects, even if there is only one document.
Each object must conform exactly to this schema:

{schema_block}

Rules:
- Respond ONLY with valid JSON. No markdown fences, no commentary.
- The top-level value MUST be a JSON array.
- If a field is not present in the document, OMIT it entirely from the output.
- Dates must be ISO 8601 (YYYY-MM-DD).
- Monetary values must be plain numbers (no currency symbols).
- For tax registration numbers, preserve the exact alphanumeric format.
- If a document contains a table of charges, extract every row as a line item.
- Do NOT use em dashes (U+2014) in any key or value; use regular hyphens.
"""


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_document(
    page_images: list[bytes],
    *,
    schema: dict[str, Any],
    model: str = _DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """
    Run the extraction pipeline on pre-rendered page images.

    Parameters
    ----------
    page_images:
        PNG byte strings for each page (produced by ``render_pdf_pages``).
    schema:
        The JSON schema definition to embed in the extraction prompt.
    model:
        Gemini model identifier (default ``gemini-2.5-flash``).

    Returns
    -------
    list[dict[str, Any]]
        One dict per logical document found in the PDF.
    """

    system_prompt = _build_system_prompt(schema)

    llm = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    content_parts: list = ["Extract structured data from this document."]
    for png in page_images:
        content_parts.append({"mime_type": "image/png", "data": png})

    logger.info("Calling %s for structured extraction", model)
    response = _call_with_retry(llm, content_parts)

    raw_text = response.text
    logger.debug("Raw LLM response: %s", raw_text)

    cleaned = _strip_json_fences(raw_text)
    payload = json.loads(cleaned)

    if isinstance(payload, dict):
        payload = [payload]

    payload = [_strip_nulls(_sanitize_em_dashes(doc)) for doc in payload]

    logger.info(
        "Extraction complete -- %d document(s) extracted",
        len(payload),
    )
    return payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_nulls(obj: Any) -> Any:
    """Recursively remove keys whose values are None, empty dicts, or empty lists."""

    if isinstance(obj, dict):
        return {
            k: _strip_nulls(v)
            for k, v in obj.items()
            if v is not None and v != {} and v != []
        }
    if isinstance(obj, list):
        return [_strip_nulls(item) for item in obj if item is not None]
    return obj


def _sanitize_em_dashes(obj: Any) -> Any:
    """Recursively replace em/en dashes with regular hyphens."""

    if isinstance(obj, str):
        return obj.replace("\u2014", "-").replace("\u2013", "-")
    if isinstance(obj, dict):
        return {
            k.replace("\u2014", "-").replace("\u2013", "-"): _sanitize_em_dashes(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_sanitize_em_dashes(item) for item in obj]
    return obj


def _call_with_retry(
    llm: genai.GenerativeModel,
    content_parts: list,
) -> genai.types.GenerateContentResponse:
    """Call generate_content with exponential backoff on rate-limit errors."""

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return llm.generate_content(content_parts)
        except ResourceExhausted:
            if attempt == _MAX_RETRIES:
                raise
            delay = _BASE_RETRY_DELAY * attempt
            logger.warning(
                "Rate-limited (attempt %d/%d) -- retrying in %ds",
                attempt,
                _MAX_RETRIES,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable")


def _strip_json_fences(text: str) -> str:
    """Remove optional ```json ... ``` wrappers from LLM output."""

    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.index("\n")
        stripped = stripped[first_newline + 1 :]
    if stripped.endswith("```"):
        stripped = stripped[: -3]
    return stripped.strip()
