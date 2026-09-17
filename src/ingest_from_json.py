"""
DVC stage 2: data/processed/articles.json -> Qdrant collection.

Reads the extracted articles (produced by extract_to_json.py) and embeds
+ stores them, exactly like the /ingest endpoint does, but driven by DVC
so it re-runs automatically whenever articles.json or the embedding
config changes.
"""

import json
from pathlib import Path

from src.chunk_embed import embed_and_store
from src.config import settings
from src.schemas import Article

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_PATH = PROJECT_ROOT / "data" / "processed" / "articles.json"
RECEIPT_PATH = PROJECT_ROOT / "data" / "processed" / "ingest_receipt.json"


def main() -> None:
    with open(ARTICLES_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    articles = [Article(**item) for item in raw]
    embedded_count = embed_and_store(articles)

    receipt = {
        "total_articles": len(articles),
        "repealed_articles": sum(1 for a in articles if a.is_repealed),
        "embedded_articles": embedded_count,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "collection_name": settings.collection_name,
    }

    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RECEIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)

    print(f"[ingest] {receipt}")


if __name__ == "__main__":
    main()
