"""
FastAPI application for the Claims Document Intelligence Pipeline.

Provides a single endpoint that accepts a PDF upload, discovers the
appropriate extraction schema, runs the Gemini extraction engine, and
returns structured JSON suitable for downstream analytics, dashboards,
and audit trails.

Architecture notes (Compass Pillars -- Modern Data Foundation)
--------------------------------------------------------------
SEARCH  : POST /process-document results can be indexed into a vector store
          alongside the raw page embeddings.  A companion GET /search
          endpoint would accept natural-language queries and return ranked
          document matches via RAG (Retrieval-Augmented Generation), enabling
          legal analysts to find relevant claims without keyword guessing.

AUTOMATION : In a scaled deployment this API sits behind an event-driven
             ingestion layer.  New documents landing in cloud storage
             (S3 / Azure Blob / GCS) trigger an orchestrator (Airflow,
             Prefect, or Dagster) that calls this endpoint, validates the
             schema, and loads results into a data warehouse.  Dead-letter
             queues capture failures for manual triage.

INSIGHTS : The ProcessingResult JSON is warehouse-ready.  Flatten it into
           fact/dimension tables and connect a BI tool (Power BI, Tableau,
           Metabase) for dashboards tracking claim volumes, average amounts,
           overdue invoices, and tax-compliance metrics -- all refreshed on
           each pipeline run.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.discovery import discover_schema
from src.extractor import extract_document, render_pdf_pages
from src.models import ProcessingResult

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Logging -- provides the "Audit Trail" required by the legal firm.
# Every request is tagged with a unique request_id so that individual
# extractions can be traced end-to-end through log aggregation systems
# (e.g. ELK, Datadog, Grafana Loki).
# ---------------------------------------------------------------------------

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("doc_intel.api")

_DEFAULT_MODEL = "gemini-2.5-flash"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure the Gemini API key on startup."""
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    logger.info("Gemini API configured (model=%s) -- pipeline ready", _DEFAULT_MODEL)
    yield
    logger.info("Shutting down pipeline")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Claims Document Intelligence Pipeline",
    summary="Automated extraction of structured metadata from legal-claims documents.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.post(
    "/process-document",
    response_model=ProcessingResult,
    summary="Extract structured data from a PDF document",
)
async def process_document(file: UploadFile = File(...)):
    """
    Accept a PDF upload, discover the extraction schema, run Gemini
    vision extraction, and return a ProcessingResult containing the
    extracted metadata and the schema that was applied.
    """

    request_id = uuid.uuid4().hex
    logger.info("[%s] Received file: %s", request_id, file.filename)

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        logger.warning(
            "[%s] Rejected -- unsupported content type: %s",
            request_id,
            file.content_type,
        )
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Upload a PDF.",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("[%s] PDF size: %d bytes", request_id, len(pdf_bytes))

    # -- Step 1: Render pages (shared by discovery and extraction) -----------
    try:
        page_images = render_pdf_pages(pdf_bytes)
    except Exception:
        logger.exception("[%s] PDF rendering failed", request_id)
        raise HTTPException(status_code=422, detail="Could not render PDF pages.")

    page_count = len(page_images)
    logger.info("[%s] Rendered %d page(s)", request_id, page_count)

    # -- Step 2: Schema discovery -------------------------------------------
    try:
        schema_info = discover_schema(page_images, model=_DEFAULT_MODEL)
    except Exception:
        logger.exception("[%s] Schema discovery failed", request_id)
        raise HTTPException(
            status_code=502,
            detail="Schema discovery failed. Check server logs for details.",
        )

    logger.info(
        "[%s] Schema resolved -- type=%s, source=%s",
        request_id,
        schema_info.document_type,
        schema_info.source,
    )

    # -- Step 3: Adaptive extraction ----------------------------------------
    try:
        documents = extract_document(
            page_images,
            schema=schema_info.definition,
            model=_DEFAULT_MODEL,
        )
    except Exception:
        logger.exception("[%s] Extraction failed", request_id)
        raise HTTPException(
            status_code=502,
            detail="Upstream LLM extraction failed. Check server logs for details.",
        )

    # -- Step 4: Build response ---------------------------------------------
    result = ProcessingResult(
        request_id=request_id,
        filename=file.filename or "unknown.pdf",
        page_count=page_count,
        metadata=documents,
        schema_used=schema_info,
        model_used=_DEFAULT_MODEL,
    )

    logger.info(
        "[%s] Extraction succeeded -- %d document(s), schema=%s (%s)",
        request_id,
        len(documents),
        schema_info.document_type,
        schema_info.source,
    )

    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
