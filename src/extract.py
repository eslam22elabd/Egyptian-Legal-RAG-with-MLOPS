import re
from collections import defaultdict

import pdfplumber

from src.schemas import Article


# Matches the first real structural marker in the English column.
# Everything before this line is the issuance decree preamble and is skipped.
_SECTION_START_RE = re.compile(
    r'^(SECTION|CHAPTER|BOOK|TITLE|PART)\s*[IVXLCM0-9]*\.?\s*$', re.IGNORECASE
)


def _find_section_start(left_lines: list[tuple[int, float, str]]) -> int:
    """Return the index of the first SECTION/CHAPTER/BOOK header in the
    English column. Everything before this index is preamble and is ignored.
    Returns 0 if no such header is found (safe fallback)."""
    for idx, (_page, _top, text) in enumerate(left_lines):
        if _SECTION_START_RE.match(text.strip()):
            return idx
    return 0


# ---------------------------------------------------------------------------
# STEP 1 – Extract two clean, column-separated text streams from the PDF
# ---------------------------------------------------------------------------

DIGIT_RE = re.compile(r'^[\u0660-\u06690-9\(\)\.\,]+$')
EASTERN_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')


def _fix_token(tok: str) -> str:
    """Reverse an Arabic word's characters into logical reading order.
    Numeric tokens are left untouched since digits are already stored
    left-to-right in this PDF even inside RTL text runs."""
    if DIGIT_RE.match(tok):
        return tok
    return tok[::-1]


def _extract_columns(pdf_path: str):
    """Returns (left_lines, right_lines): each is a list of
    (page_number, top_y, line_text) tuples, in reading order."""
    left_lines: list[tuple[int, float, str]] = []
    right_lines: list[tuple[int, float, str]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages):
            mid = page.width / 2
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False)

            by_line: dict[float, list] = defaultdict(list)
            for w in words:
                by_line[round(w['top'], 0)].append(w)

            for top in sorted(by_line):
                ws = by_line[top]
                left_ws = [w for w in ws if w['x0'] < mid]
                right_ws = [w for w in ws if w['x0'] >= mid]

                if left_ws:
                    left_ws_sorted = sorted(left_ws, key=lambda w: w['x0'])
                    left_text = ' '.join(w['text'] for w in left_ws_sorted)
                    left_lines.append((pno, top, left_text))

                if right_ws:
                    # RTL: rightmost word is read first
                    right_ws_sorted = sorted(right_ws, key=lambda w: -w['x0'])
                    right_text = ' '.join(_fix_token(w['text']) for w in right_ws_sorted)
                    right_lines.append((pno, top, right_text))

    return left_lines, right_lines


# ---------------------------------------------------------------------------
# STEP 2 – Slice each column into individual articles
# ---------------------------------------------------------------------------

EN_ARTICLE_RE = re.compile(r'^Article\s+(\d+)\b')
AR_ARTICLE_RE = re.compile(r'^مادة[\s\(\)]*([٠-٩0-9]+)')

REPEALED_PATTERN = re.compile(
    r"Articles?\s+(\d+)(?:\s*-\s*(\d+))?\s+ha(?:s|ve) been repealed",
    re.IGNORECASE,
)


def _find_article_headers(
    lines: list[tuple[int, float, str]],
    pattern: re.Pattern,
    is_arabic: bool = False,
) -> list[dict]:
    """Find every line that starts a new article.
    Returns [{'num': int, 'idx': line_index, 'page': page_number}, ...]"""
    heads = []
    for idx, (page, _top, text) in enumerate(lines):
        m = pattern.match(text.strip())
        if not m:
            continue
        num_str = m.group(1).translate(EASTERN_DIGITS) if is_arabic else m.group(1)
        try:
            num = int(num_str)
        except ValueError:
            continue
        heads.append({'num': num, 'idx': idx, 'page': page})
    return heads


def _slice_articles(
    lines: list[tuple[int, float, str]],
    heads: list[dict],
    skip_header_lines: int = 1,
) -> dict[int, dict]:
    """For each header, join all lines up to the next header into that
    article's body text. Keeps the first occurrence if a number repeats
    (guards against stray false-positive matches)."""
    out: dict[int, dict] = {}
    for i, h in enumerate(heads):
        start = h['idx'] + skip_header_lines
        end = heads[i + 1]['idx'] if i + 1 < len(heads) else len(lines)
        text = '\n'.join(lines[j][2] for j in range(start, end)).strip()
        if h['num'] not in out:
            out[h['num']] = {'text': text, 'page': h['page']}
    return out


# ---------------------------------------------------------------------------
# STEP 3 – Build the table of contents (Book / Chapter / Section hierarchy)
# ---------------------------------------------------------------------------

def _eng_part(s: str) -> str:
    """Return only the leading ASCII portion of a mixed-language line."""
    m = re.match(r'^([\x00-\x7F]+)', s)
    return m.group(1).strip() if m else ''


def _build_section_map(
    left_lines: list[tuple[int, float, str]],
    en_article_positions: list[tuple[int, int]],
) -> list[dict]:
    """Detect top-level headers (SECTION / CHAPTER / BOOK / PART / TITLE)
    in the English column and compute which article-number range falls
    under each one."""
    headers: list[tuple[int, str, str]] = []
    texts = [t for (_, _, t) in left_lines]

    for i, text in enumerate(texts):
        left = _eng_part(text)
        if not left:
            continue
        if re.match(r'^(SECTION|CHAPTER|BOOK|TITLE|PART)\s*[IVXLCM0-9]*\.?\s*$', left, re.I):
            title = ''
            for j in range(i + 1, min(i + 3, len(texts))):
                nxt = _eng_part(texts[j])
                if nxt and not re.match(r'^(Article|\d+\.)', nxt) and \
                   not re.match(r'^(SECTION|CHAPTER|BOOK|TITLE|PART)\b', nxt, re.I):
                    title = nxt
                break
            headers.append((i, left, title))

    sections: list[dict] = []
    for idx, (line_idx, code, title) in enumerate(headers):
        if not title:
            continue
        end_idx = headers[idx + 1][0] if idx + 1 < len(headers) else 10 ** 9
        arts = sorted(a for (li, a) in en_article_positions if line_idx < li < end_idx)
        if arts:
            sections.append({'code': code, 'title': title, 'first': arts[0], 'last': arts[-1]})
    return sections


def _find_section_for_article(sections: list[dict], num: int) -> str | None:
    for s in sections:
        if s['first'] <= num <= s['last']:
            return s['title']
    return None


# ---------------------------------------------------------------------------
# STEP 4 – Cross-references
# ---------------------------------------------------------------------------

def _cross_references(text: str, self_num: int) -> list[int]:
    refs = set(int(n) for n in re.findall(r'Article\s+(\d+)', text))
    refs.discard(self_num)
    return sorted(refs)


# ---------------------------------------------------------------------------
# Public API (same signature as before)
# ---------------------------------------------------------------------------

def parse_articles(pdf_path: str) -> list[Article]:
    """Extract bilingual Article records from the two-column Egyptian Civil
    Code PDF. Returns one Article per article number with both English and
    Arabic text, section hierarchy, cross-references, and page number.

    The PDF starts with an issuance-decree preamble (مادة ١ / مادة ٢) that
    has different content from the actual civil-code Article 1 / Article 2.
    We skip everything before the first SECTION marker in the English column
    to avoid merging preamble Arabic text with civil-code English text.
    """
    left_lines, right_lines = _extract_columns(pdf_path)

    # --- Preamble trim ---------------------------------------------------
    # Find the page where the first SECTION header appears in the English
    # column. Both columns are trimmed to that page so the preamble Arabic
    # articles (مادة ١, مادة ٢) are never seen by the article parser.
    section_start_idx = _find_section_start(left_lines)
    if section_start_idx > 0:
        start_page = left_lines[section_start_idx][0]
        start_top = left_lines[section_start_idx][1]
        left_lines = left_lines[section_start_idx:]
        # Use page + y-position so Arabic preamble lines on the *same page*
        # but *above* the first SECTION marker are also excluded.
        right_lines = [
            ln for ln in right_lines
            if ln[0] > start_page or (ln[0] == start_page and ln[1] >= start_top)
        ]
    # ---------------------------------------------------------------------

    en_heads = _find_article_headers(left_lines, EN_ARTICLE_RE, is_arabic=False)
    ar_heads = _find_article_headers(right_lines, AR_ARTICLE_RE, is_arabic=True)

    en_articles = _slice_articles(left_lines, en_heads)
    ar_articles = _slice_articles(right_lines, ar_heads)

    en_article_positions = [(h['idx'], h['num']) for h in en_heads]
    sections = _build_section_map(left_lines, en_article_positions)

    results: dict[int, Article] = {}
    # Intersection only: orphaned preamble EN-only/AR-only entries are discarded.
    all_nums = sorted(set(en_articles) & set(ar_articles))

    for num in all_nums:
        en_entry = en_articles.get(num)
        ar_entry = ar_articles.get(num)
        en_text = en_entry['text'] if en_entry else ''
        ar_text = ar_entry['text'] if ar_entry else ''
        page = en_entry['page'] if en_entry else (ar_entry['page'] if ar_entry else None)

        # Handle repealed articles
        repealed_match = REPEALED_PATTERN.search(en_text) if en_text else None
        if repealed_match:
            start_number = int(repealed_match.group(1))
            end_number = (
                int(repealed_match.group(2)) if repealed_match.group(2) else start_number
            )
            plural = start_number != end_number
            repeal_note = (
                f"Article{'s' if plural else ''} {start_number}"
                f"{'-' + str(end_number) if plural else ''} "
                f"{'have' if plural else 'has'} been repealed "
                "by Presidential Decree."
            )
            for r_num in range(start_number, end_number + 1):
                results[r_num] = Article(
                    article_number=r_num,
                    text_en=repeal_note,
                    text_ar='',
                    is_repealed=True,
                    page=page,
                )
            if start_number <= num <= end_number:
                continue

        results[num] = Article(
            article_number=num,
            text_en=en_text,
            text_ar=ar_text,
            is_repealed=False,
            section_title=_find_section_for_article(sections, num),
            cross_references=_cross_references(en_text, num),
            page=page,
        )

    return [results[n] for n in sorted(results)]