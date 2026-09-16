"""
Chunking & embedding pipeline: Article objects → Qdrant vectors.

Embedding model: BAAI/bge-m3 (multilingual, 1024-dim).
Each article is embedded as a single bilingual text block:
    Section: <section_title>
    Article N: <english_text>
    مادة N: <arabic_text>
"""

import time

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import settings
from src.schemas import Article


_model: SentenceTransformer | None = None
_client: QdrantClient | None = None

# Smaller batch → fewer vectors per upsert → less chance of Qdrant Cloud timeout
BATCH_SIZE = 16

# Retry settings for Qdrant upsert
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds (doubles each retry)


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=60,  # seconds — avoids hanging on slow Qdrant Cloud responses
        )
    return _client


def ensure_collection(client: QdrantClient) -> None:
    collections = client.get_collections().collections

    if not any(c.name == settings.collection_name for c in collections):
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )


def article_to_text(article: Article) -> str:
    """
    Build the bilingual embedding text for an article.

    Feeds both languages + the section title into bge-m3 so the encoder
    captures topical context beyond the bare article text.
    Arabic questions will match on the Arabic portion; English questions
    on the English portion — same vector space thanks to bge-m3.
    """
    parts: list[str] = []
    if article.section_title:
        parts.append(f"Section: {article.section_title}")
    if article.text_en and article.text_en.strip():
        parts.append(article.text_en.strip())
    if article.text_ar and article.text_ar.strip():
        parts.append(f"مادة {article.article_number}:\n{article.text_ar.strip()}")
    return "\n\n".join(parts).strip()


def _chunked(seq: list[Article], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def embed_and_store(
    articles: list[Article],
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    Create one embedding per non-repealed article and store it in Qdrant.

    Point ID = article number, so direct lookups such as 'المادة 1'
    can retrieve the exact Qdrant point using that number.
    """
    non_repealed = [article for article in articles if not article.is_repealed]

    if not non_repealed:
        return 0

    model = get_embedding_model()
    client = get_qdrant_client()
    ensure_collection(client)

    total_stored = 0
    batches = list(_chunked(non_repealed, batch_size))

    for batch_index, batch in enumerate(batches, start=1):
        texts = [article_to_text(article) for article in batch]

        vectors = model.encode(
            texts,
            convert_to_numpy=True,
        ).tolist()

        points = [
            PointStruct(
                id=article.article_number,
                vector=vector,
                payload=article.model_dump(),
            )
            for article, vector in zip(batch, vectors)
        ]

        # Retry upsert on transient Qdrant Cloud timeouts
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                client.upsert(
                    collection_name=settings.collection_name,
                    points=points,
                    wait=True,
                )
                break
            except Exception as exc:
                if attempt == _MAX_RETRIES:
                    raise
                wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                print(
                    f"[ingest] upsert failed (attempt {attempt}/{_MAX_RETRIES}): {exc}\n"
                    f"         retrying in {wait:.0f}s..."
                )
                time.sleep(wait)

        total_stored += len(points)

        print(
            f"[ingest] batch {batch_index}/{len(batches)} stored "
            f"({total_stored}/{len(non_repealed)} articles so far)"
        )

    return total_stored