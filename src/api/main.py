import shutil
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.extract import parse_articles
from src.chunk_embed import embed_and_store
from src.retrieve import Retriever
from src.generate import Generator
from src.config import settings

app = FastAPI(title="Egyptian Legal RAG", version="0.1.0")

_retriever: Retriever | None = None
_generator: Generator | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["ما هي شروط صحة العقد؟"])


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    correlation_id: str
    latency_ms: float


class IngestResponse(BaseModel):
    total_articles: int
    repealed_articles: int
    embedded_articles: int


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        articles = parse_articles(tmp_path)
        embedded_count = embed_and_store(articles)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    global _retriever
    _retriever = None  # force re-init so the next /ask uses fresh data

    repealed = sum(1 for a in articles if a.is_repealed)
    return IngestResponse(
        total_articles=len(articles),
        repealed_articles=repealed,
        embedded_articles=embedded_count,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.time()
    cid = str(uuid.uuid4())
    articles = get_retriever().retrieve(req.question, top_k=settings.reranker_top_k)
    result = get_generator().generate(req.question, articles)
    latency_ms = (time.time() - start) * 1000
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        correlation_id=cid,
        latency_ms=latency_ms,
    )
