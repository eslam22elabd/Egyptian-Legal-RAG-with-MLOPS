"""
Retriever: query -> top-k relevant articles from Qdrant Cloud.

Pipeline:
  1. Direct lookup   — if the question explicitly mentions an article number,
                       fetch that article by exact Qdrant point ID.
  2. Semantic search — embed the query with bge-m3, ANN search in Qdrant,
                       retrieve top_k * RERANK_FETCH_FACTOR candidates.
  3. Rerank          — score (query, article_text) pairs with bge-reranker-v2-m3
                       and keep only the top reranker_top_k results.

Direct-lookup hits are always included and are not reranked (they are exact
matches by definition). The reranker only reorders the semantic candidates.
"""

import re

from sentence_transformers import CrossEncoder

from src.chunk_embed import get_embedding_model, get_qdrant_client
from src.config import settings

# How many semantic candidates to fetch before reranking.
# A larger pool gives the reranker more to work with.
RERANK_FETCH_FACTOR = 4

# Regex to match "Article 60", "article no. 60", "المادة 60", "مادة 60"
ARTICLE_NUMBER_PATTERN = re.compile(
    r"(?:articles?|مواد|المواد|مادة|المادة)\s*(?:no\.?|رقم)?\s*([٠-٩0-9]+)",
    re.IGNORECASE,
)

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def _extract_article_numbers(question: str) -> list[int]:
    matches = ARTICLE_NUMBER_PATTERN.finditer(question)
    return [int(m.group(1)) for m in matches]


class Retriever:
    def __init__(self) -> None:
        self.model = get_embedding_model()
        self.client = get_qdrant_client()
        self.reranker = get_reranker()

    def _direct_lookup(self, article_number: int) -> dict | None:
        """Fetch a single article by its exact ID from Qdrant."""
        points = self.client.retrieve(
            collection_name=settings.collection_name,
            ids=[article_number],
        )
        return points[0].payload if points else None

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        vec = self.model.encode([query], convert_to_numpy=True)[0].tolist()
        results = self.client.query_points(
            collection_name=settings.collection_name, query=vec, limit=top_k
        )
        return [r.payload for r in results.points]

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Score each candidate with the cross-encoder and return them
        sorted by descending relevance score."""
        if not candidates:
            return candidates

        reranker = self.reranker

        # Build (query, passage) pairs — use the same bilingual text that
        # was used during indexing so the reranker sees the same content.
        def _candidate_text(payload: dict) -> str:
            parts: list[str] = []
            if payload.get("section_title"):
                parts.append(f"Section: {payload['section_title']}")
            if payload.get("text_en"):
                parts.append(payload['text_en'])
            if payload.get("text_ar"):
                parts.append(
                    f"مادة {payload['article_number']}:\n{payload['text_ar']}"
                )
            return "\n\n".join(parts).strip()

        pairs = [(query, _candidate_text(c)) for c in candidates]
        scores = reranker.predict(pairs)

        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    def retrieve(self, query: str, top_k: int = settings.reranker_top_k) -> list[dict]:
        results: list[dict] = []
        seen_numbers: set[int] = set()

        # 1. Direct lookup — pinned, not reranked
        article_numbers = _extract_article_numbers(query)
        for num in article_numbers:
            direct = self._direct_lookup(num)
            if direct is not None and direct["article_number"] not in seen_numbers:
                results.append(direct)
                seen_numbers.add(direct["article_number"])

        # 2. Semantic search with an expanded candidate pool
        fetch_k = top_k * RERANK_FETCH_FACTOR + len(article_numbers)
        semantic_candidates = [
            p for p in self._semantic_search(query, top_k=fetch_k)
            if p["article_number"] not in seen_numbers
        ]

        # 3. Rerank semantic candidates
        reranked = self._rerank(query, semantic_candidates)

        # Fill up to top_k with reranked results (direct lookups already counted)
        remaining = max(top_k - len(results), 0)
        for payload in reranked[:remaining]:
            if payload["article_number"] not in seen_numbers:
                results.append(payload)
                seen_numbers.add(payload["article_number"])

        return results