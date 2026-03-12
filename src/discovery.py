"""
Dynamic Schema Discovery for the Claims Document Intelligence Pipeline.

Classifies incoming documents and, for previously unseen types, generates
a bespoke JSON schema on-the-fly via LLM.  Discovered schemas are persisted
in a local registry file so that future documents of the same type are
extracted with a consistent structure.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from src.models import SchemaInfo

logger = logging.getLogger("doc_intel.discovery")

_MAX_RETRIES = 3
_BASE_RETRY_DELAY = 10

REGISTRY_PATH = Path("schema_registry.json")

# ---------------------------------------------------------------------------
# Predefined schemas for known document types
# ---------------------------------------------------------------------------

PREDEFINED_SCHEMAS: dict[str, dict[str, Any]] = {
    "invoice": {
        "document_type": "string",
        "client": {
            "name": "string",
            "address": "string",
            "contact_email": "string",
        },
        "vendor": {
            "name": "string",
            "address": "string",
            "contact_email": "string",
        },
        "invoice_number": "string",
        "purchase_order": "string",
        "invoice_date": "date (YYYY-MM-DD)",
        "due_date": "date (YYYY-MM-DD)",
        "currency": "string (ISO 4217)",
        "subtotal": "number",
        "total_amount": "number",
        "amount_due": "number",
        "tax": {
            "gst_number": "string",
            "qst_number": "string",
            "tax_rate_percent": "number",
            "tax_amount": "number",
        },
        "line_items": [
            {
                "description": "string",
                "quantity": "number",
                "unit_price": "number",
                "amount": "number",
            }
        ],
        "additional_metadata": "object",
    },
    "contract": {
        "document_type": "string",
        "title": "string",
        "parties": [
            {
                "name": "string",
                "role": "string",
                "address": "string",
            }
        ],
        "contract_number": "string",
        "effective_date": "date (YYYY-MM-DD)",
        "expiration_date": "date (YYYY-MM-DD)",
        "contract_value": "number",
        "currency": "string (ISO 4217)",
        "key_terms": ["string"],
        "governing_law": "string",
        "additional_metadata": "object",
    },
    "meeting_minutes": {
        "document_type": "string",
        "meeting_title": "string",
        "meeting_date": "date (YYYY-MM-DD)",
        "location": "string",
        "attendees": [
            {
                "name": "string",
                "role": "string",
            }
        ],
        "agenda_items": [
            {
                "topic": "string",
                "discussion": "string",
                "decision": "string",
            }
        ],
        "action_items": [
            {
                "assignee": "string",
                "task": "string",
                "due_date": "date (YYYY-MM-DD)",
            }
        ],
        "additional_metadata": "object",
    },
}

_KNOWN_TYPES = frozenset(PREDEFINED_SCHEMAS.keys())

# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

_CLASSIFICATION_PROMPT = """\
You are a document classification and schema generation agent.

Analyse the provided document image and determine its type.

Known document types (use these exact names if the document matches):
  "invoice", "contract", "meeting_minutes"

Respond with a JSON object:

If the document matches a KNOWN type:
  {"document_type": "<known_type>", "is_known": true}

If the document does NOT match any known type, also generate a JSON schema
that best captures the document's unique fields:
  {
    "document_type": "<descriptive_snake_case_name>",
    "is_known": false,
    "schema": {
      "<field_name>": "<data_type>",
      ...
    }
  }

Schema rules:
- Use lowercase snake_case for all field names.
- Valid data types: "string", "number", "date (YYYY-MM-DD)", "boolean", "object".
- For arrays use a list containing one example element, e.g. [{"name": "string"}].
- Always include a "document_type": "string" field.
- Always include an "additional_metadata": "object" field as a catch-all.
- Do NOT use em dashes (--) in any key or value.
- Respond ONLY with valid JSON. No markdown fences, no commentary.
"""


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def load_registry() -> dict[str, dict[str, Any]]:
    """Load the schema registry from disk, returning an empty dict if missing."""
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(registry: dict[str, dict[str, Any]]) -> None:
    """Persist the schema registry to disk."""
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Schema registry saved (%d custom type(s))", len(registry))


# ---------------------------------------------------------------------------
# Core discovery function
# ---------------------------------------------------------------------------

def discover_schema(
    page_images: list[bytes],
    *,
    model: str = "gemini-2.5-flash",
) -> SchemaInfo:
    """
    Classify a document and resolve the extraction schema.

    Sends the first page to the LLM for classification.  For known types
    the predefined schema is returned immediately.  For unknown types the
    registry is checked; if absent the LLM generates a new schema which is
    then persisted.

    Parameters
    ----------
    page_images:
        Pre-rendered PNG bytes for each page (only the first is sent).
    model:
        Gemini model identifier.

    Returns
    -------
    SchemaInfo with document_type, source, and definition.
    """

    llm = genai.GenerativeModel(
        model_name=model,
        system_instruction=_CLASSIFICATION_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    content_parts: list = [
        "Classify this document and generate a schema if needed.",
        {"mime_type": "image/png", "data": page_images[0]},
    ]

    response = _call_with_retry(llm, content_parts)
    raw = _strip_json_fences(response.text)
    result = json.loads(raw)

    doc_type: str = result["document_type"].lower().strip()
    is_known: bool = result.get("is_known", doc_type in _KNOWN_TYPES)

    # --- Known / predefined type ---
    if is_known and doc_type in _KNOWN_TYPES:
        logger.info(
            "AUDIT -- schema source=predefined, document_type=%s", doc_type
        )
        return SchemaInfo(
            document_type=doc_type,
            source="predefined",
            definition=PREDEFINED_SCHEMAS[doc_type],
        )

    # --- Unknown type: check registry first ---
    registry = load_registry()

    if doc_type in registry:
        logger.info(
            "AUDIT -- schema source=reused (from registry), document_type=%s",
            doc_type,
        )
        return SchemaInfo(
            document_type=doc_type,
            source="reused",
            definition=registry[doc_type],
        )

    # --- Newly discovered type: persist to registry ---
    schema_def: dict[str, Any] = result.get("schema", {})
    if "document_type" not in schema_def:
        schema_def["document_type"] = "string"
    if "additional_metadata" not in schema_def:
        schema_def["additional_metadata"] = "object"

    registry[doc_type] = schema_def
    save_registry(registry)

    logger.info(
        "AUDIT -- schema source=discovered (NEW), document_type=%s, "
        "fields=%s",
        doc_type,
        list(schema_def.keys()),
    )

    return SchemaInfo(
        document_type=doc_type,
        source="discovered",
        definition=schema_def,
    )


# ---------------------------------------------------------------------------
# Shared helpers (mirror of extractor utilities)
# ---------------------------------------------------------------------------

def _call_with_retry(
    llm: genai.GenerativeModel,
    content_parts: list,
) -> genai.types.GenerateContentResponse:
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
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.index("\n")
        stripped = stripped[first_newline + 1 :]
    if stripped.endswith("```"):
        stripped = stripped[: -3]
    return stripped.strip()
