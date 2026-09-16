"""
Generator: question + retrieved articles -> answer with citations.

Retrieval evidence is bilingual (EN + AR) thanks to the new pdfplumber
extraction and bge-m3 embeddings. Gemini receives both language versions
of each article and is asked to answer in the user's question language.
"""

from google import genai

from src.config import settings
from src.schemas import Article


PROMPT_TEMPLATE = """You are a legal assistant answering questions about the Egyptian Civil Code.

Use ONLY the provided article excerpts as evidence.
If the excerpts do not contain enough information to answer the question,
say clearly that the provided articles do not contain the answer.
Do not invent legal rules.
Always cite the article number or article numbers used.

The article excerpts are provided in both English and Arabic.
Answer in the same language as the user's question.

Question:
{question}

Relevant articles:
{context}

Answer:
"""


class Generator:
    def __init__(self) -> None:
        self._ready = False
        self._client = None

        if settings.gemini_api_key:
            self._client = genai.Client(api_key=settings.gemini_api_key)
            self._ready = True

    def generate(self, query: str, articles: list[dict]) -> dict:
        if not articles:
            return {
                "answer": "No relevant articles found.",
                "sources": [],
            }

        parsed_articles = [Article(**article) for article in articles]
        sources = [article.citation for article in parsed_articles]

        if not self._ready:
            top_article = parsed_articles[0]
            return {
                "answer": top_article.text_en or top_article.text_ar,
                "sources": sources,
            }

        context_parts: list[str] = []
        for article in parsed_articles:
            header = f"[{article.citation}]"
            en_body = article.text_en or ""
            ar_body = article.text_ar or ""
            if en_body and ar_body:
                context_parts.append(f"{header}\n{en_body}\n\n{ar_body}")
            else:
                context_parts.append(f"{header}\n{en_body or ar_body}")
        context = "\n\n".join(context_parts)

        prompt = PROMPT_TEMPLATE.format(
            question=query,
            context=context,
        )

        response = self._client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )

        return {
            "answer": response.text,
            "sources": sources,
        }