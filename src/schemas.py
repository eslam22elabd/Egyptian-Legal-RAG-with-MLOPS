"""
Data model for one article of the Egyptian Civil Code.
"""

from pydantic import BaseModel


class Article(BaseModel):
    article_number: int
    text_ar: str = ""
    text_en: str | None = None
    is_repealed: bool = False

    # Bilingual extraction metadata
    section_title: str | None = None
    cross_references: list[int] = []
    page: int | None = None

    @property
    def citation(self) -> str:
        return f"Egyptian Civil Code, Article {self.article_number}"
