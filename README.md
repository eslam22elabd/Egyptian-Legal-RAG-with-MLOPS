# Egyptian Legal RAG

RAG service over the Egyptian Civil Code (bilingual Arabic/English PDF).
No Docker — Qdrant Cloud is used directly as the vector store API.

## Setup
```bash
py -3.14 -m venv .venv
.venv\Scripts\Activate      # Windows
# source .venv/bin/activate  # Linux/Mac

pip install --upgrade pip
pip install -e ".[dev]"
```

`.env` already has your Qdrant Cloud URL + API key filled in. You still
need to add your Gemini API key (free at https://aistudio.google.com/app/apikey).

## Run the API
```bash
cd src/api
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to try it from the browser.

## Workflow
1. `POST /ingest` — upload the PDF (multipart/form-data, field name `file`).
   The server extracts articles, chunks (1 article = 1 chunk), embeds them,
   and upserts into your Qdrant Cloud collection. Returns counts.
2. `POST /ask` — send `{"question": "..."}`, get back an answer + article
   citations, generated via Gemini using the retrieved articles as context.

Example:
```bash
curl -X POST http://127.0.0.1:8000/ingest -F "file=@../../data/raw/egyptian_civil_code_bilingual.pdf"

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"ما هي شروط صحة العقد؟\"}"
```

## Extraction logic (validated against your real PDF)
The PDF is two side-by-side columns (Arabic + English) per page. PyMuPDF's
`get_text("blocks")` gives each block's bounding box; we split blocks into
LEFT/RIGHT columns by x-position, then decide which side is Arabic by
checking which one actually contains more Arabic characters (verified per
document, not assumed).

Key finding from your actual file: Arabic text comes out in CORRECT letter
order with this method (unlike plain pdfplumber text extraction, which
reversed Arabic-letter runs) — no character-repair step needed.

The two columns are NOT guaranteed to align on the same page (translation
length differs), so we build two independent global streams (all LEFT
blocks across all 170 pages, all RIGHT blocks across all pages) and pair
articles by their sequence position, not by page. English "Article N"
lines are the reliable numbering anchor; Arabic marker positions are used
only to split into segments in order.

Repealed ranges ("Articles 54-80 have been repealed...") are expanded into
one flagged record per missing number (`is_repealed=True`), never silently
dropped.

## Tests
```bash
pytest -v
```
Tests mock Qdrant/Gemini, so they run offline without needing your real
PDF, cloud cluster, or API key.

## Structure
```
egyptian-legal-rag/
├── pyproject.toml
├── .env                  
├── .gitignore
gets written here on /ingest
├── src/
│   ├── config.py
│   ├── schemas.py
│   ├── extract.py           # PDF ->Articlelist
│   ├── chunk_embed.py        # chunking + embeddings + Qdrant upsert
│   ├── retrieve.py            # query -> top-k articles
│   ├── generate.py             # Gemini-based answer generation
│   └── api/
│       └── main.py             # FastAPI: /health /ingest /ask
└── tests/
    ├── test_extract.py
    └── test_api.py
```
