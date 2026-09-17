"""
DVC stage 1: PDF -> data/processed/articles.json

Wraps src.extract.parse_articles so DVC can track it as a reproducible
pipeline stage with declared deps/outs.
"""

import json
from pathlib import Path

from src.extract import parse_articles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "egyptian_civil_code_bilingual.pdf"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "articles.json"


def main() -> None:
    articles = parse_articles(str(PDF_PATH))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [a.model_dump() for a in articles],
            f,
            ensure_ascii=False,
            indent=2,
        )

    repealed = sum(1 for a in articles if a.is_repealed)
    print(
        f"[extract] {len(articles)} articles written to {OUTPUT_PATH} "
        f"({repealed} repealed)"
    )


if __name__ == "__main__":
    main()
