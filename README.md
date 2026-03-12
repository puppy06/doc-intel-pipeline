# Claims Document Intelligence Pipeline

Automated ingestion and structured extraction of legal-claims documents
(invoices, contracts, emails) using Google Gemini as a Vision-Language Model.

## Quick start

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your Google API key
cp .env.example .env
# Edit .env and paste your key from https://aistudio.google.com/apikey

# 4. Run the API server
uvicorn src.main:app --reload
```

The server starts at `http://127.0.0.1:8000`.
Interactive docs are available at `http://127.0.0.1:8000/docs`.

## Usage

### Swagger UI (easiest)

Open `http://127.0.0.1:8000/docs` in your browser, expand
**POST /process-document**, click **Try it out**, choose a PDF file,
and hit **Execute**.

### PowerShell (Windows)

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/process-document `
  -Method Post `
  -Form @{ file = Get-Item .\data\case_interview_dataset.pdf } |
  ConvertTo-Json -Depth 10
```

### curl (macOS / Linux)

```bash
curl -X POST http://127.0.0.1:8000/process-document \
  -F "file=@data/case_interview_dataset.pdf"
```

### curl (Windows CMD)

```cmd
curl -X POST http://127.0.0.1:8000/process-document ^
  -F "file=@data\case_interview_dataset.pdf"
```

The response is a JSON object conforming to the `ProcessingResult` schema
(see `src/models.py`), containing:

- **`metadata`** -- extracted data (client names, invoice amounts, dates, tax
  registration numbers, line items, etc.)
- **`schema_used`** -- the schema that was applied (predefined, discovered,
  or reused from the registry)

## Project structure

```
src/
  models.py      -- Pydantic schemas (SchemaInfo, ProcessingResult, ...)
  discovery.py   -- Dynamic schema classification, generation, and registry
  extractor.py   -- Gemini vision extraction engine (accepts any schema)
  main.py        -- FastAPI application with POST /process-document
data/            -- Sample PDFs for testing
requirements.txt -- Python dependencies
.env.example     -- Template for environment variables
```

`schema_registry.json` is created at runtime when the pipeline encounters
a document type outside the three built-in types (invoice, contract,
meeting_minutes). It is git-ignored.

## Scaling and extensibility (Compass Pillars)

Detailed architectural notes are embedded as module-level docstrings in
`src/extractor.py` and `src/main.py`. A summary:

| Pillar         | Approach                                                                                     |
| -------------- | -------------------------------------------------------------------------------------------- |
| **Search**     | Embed extracted chunks into a vector DB (Pinecone, Weaviate, pgvector) for RAG-based Q&A.    |
| **Automation** | Wrap the extraction call in an Airflow/Prefect/Dagster DAG for end-to-end ETL orchestration. |
| **Insights**   | Flatten JSON into star-schema tables; connect Power BI / Tableau for live dashboards.        |
