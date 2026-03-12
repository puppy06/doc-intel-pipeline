"""
Pydantic schemas for the Document Intelligence Pipeline.

These models enforce strict typing so that extracted data is immediately
ready for downstream querying (SQL/DataFrame joins) and audit-trail
record-keeping.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    INVOICE = "invoice"
    CONTRACT = "contract"
    EMAIL = "email"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Nested value objects (kept for reference / optional strict validation)
# ---------------------------------------------------------------------------

class LineItem(BaseModel):
    """Single billable line inside an invoice or contract."""

    description: str = Field(..., description="Free-text description of the item or service")
    quantity: float | None = Field(None, description="Number of units (may be absent)")
    unit_price: float | None = Field(None, description="Price per unit before tax")
    amount: float = Field(..., description="Total amount for this line item")


class TaxRegistration(BaseModel):
    """Tax registration identifiers (Canadian GST/QST or equivalent)."""

    gst_number: str | None = Field(None, description="GST / HST registration number")
    qst_number: str | None = Field(None, description="QST registration number (Quebec)")
    tax_rate_percent: float | None = Field(None, description="Effective tax rate as a percentage")
    tax_amount: float | None = Field(None, description="Absolute tax amount")


class PartyInfo(BaseModel):
    """Represents a named party (client, vendor, or counterparty)."""

    name: str | None = Field(None, description="Legal or trade name")
    address: str | None = Field(None, description="Mailing or registered address")
    contact_email: str | None = Field(None, description="Primary contact email")


class ExtractedDocument(BaseModel):
    """
    Canonical output schema returned by the extraction engine for known
    document types (invoice, contract, meeting_minutes).

    Every field is nullable so that the model gracefully handles partial
    extractions -- downstream consumers decide which fields are mandatory
    for their use-case.
    """

    document_type: DocumentType = Field(
        ..., description="Classified document category"
    )

    client: PartyInfo | None = Field(None, description="The billed client / recipient")
    vendor: PartyInfo | None = Field(None, description="The issuing vendor / sender")

    invoice_number: str | None = Field(None, description="Unique invoice or reference number")
    purchase_order: str | None = Field(None, description="Associated PO number, if present")

    invoice_date: date | None = Field(None, description="Date the document was issued")
    due_date: date | None = Field(None, description="Payment due date")

    currency: str | None = Field(None, description="ISO 4217 currency code (e.g. CAD, USD)")
    subtotal: float | None = Field(None, description="Pre-tax subtotal")
    total_amount: float | None = Field(None, description="Grand total including taxes")
    amount_due: float | None = Field(None, description="Outstanding balance, if different from total")

    tax: TaxRegistration | None = Field(None, description="Tax registration and amounts")

    line_items: list[LineItem] = Field(default_factory=list, description="Itemised charges")

    additional_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra key-value pairs not captured by the fixed schema",
    )


# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------

class SchemaInfo(BaseModel):
    """Describes which schema was applied during extraction."""

    document_type: str = Field(
        ..., description="Classified document type (e.g. invoice, contract, receipt)"
    )
    source: Literal["predefined", "discovered", "reused"] = Field(
        ...,
        description=(
            "'predefined' for built-in types, 'discovered' when the LLM "
            "generated a new schema, 'reused' when a previously discovered "
            "schema was loaded from the registry"
        ),
    )
    definition: dict[str, Any] = Field(
        ..., description="The JSON schema definition (field names to data types)"
    )


# ---------------------------------------------------------------------------
# API response wrapper
# ---------------------------------------------------------------------------

class ProcessingResult(BaseModel):
    """Top-level response returned by the /process-document endpoint."""

    request_id: str = Field(..., description="Unique trace ID for audit-trail correlation")
    filename: str = Field(..., description="Original uploaded filename")
    page_count: int = Field(..., description="Number of pages processed")
    metadata: list[dict[str, Any]] = Field(
        ..., description="Extracted data -- one dict per logical document in the PDF"
    )
    schema_used: SchemaInfo = Field(
        ..., description="The schema that was applied during extraction"
    )
    processed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of processing completion",
    )
    model_used: str = Field(..., description="LLM/VLM model identifier used for extraction")
    confidence_note: str = Field(
        default="Extraction performed by LLM; manual review recommended for legal use.",
        description="Disclaimer for downstream consumers",
    )
