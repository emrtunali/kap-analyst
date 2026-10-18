"""Evaluation-label loading and verification.

Every QA item names the page(s) where its answer is printed. ``verify_labels`` checks
mechanically that the answer string really occurs on each gold page of the source PDF,
so the benchmark cannot silently drift from the documents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config
from .ingest import Document, page_texts
from .textnorm import squash_ws, tr_lower


@dataclass(frozen=True)
class QAItem:
    id: str
    doc_id: str
    lang: str
    question: str
    answer: str
    gold_pages: list[int]


def load_qa(path: Path = config.EVAL_DIR / "qa.jsonl") -> list[QAItem]:
    with path.open(encoding="utf-8") as f:
        return [QAItem(**json.loads(line)) for line in f if line.strip()]


def _norm(text: str) -> str:
    return tr_lower(squash_ws(text)).replace("( ", "(")


def answer_bearing_pages(item: QAItem, norm_pages: list[str]) -> set[int]:
    """Gold pages plus every other page that prints the same specific figure.

    Annual reports repeat headline numbers (e.g. revenue appears in the income statement
    *and* in the revenue note). A retriever that surfaces either page has found the answer,
    so for distinctive numeric answers (≥5 chars, contains a digit) any page containing the
    exact string counts as relevant. Short or textual answers keep only the hand-checked pages.
    """
    pages = set(item.gold_pages)
    needle = _norm(item.answer)
    if len(needle) >= 5 and any(c.isdigit() for c in needle):
        pages |= {i + 1 for i, p in enumerate(norm_pages) if needle in p}
    return pages


def verify_labels(items: list[QAItem], docs: list[Document]) -> list[str]:
    """Return a list of human-readable problems (empty list = all labels verified)."""
    by_id = {d.doc_id: d for d in docs}
    cache: dict[str, list[str]] = {}
    problems = []
    for it in items:
        if it.doc_id not in by_id:
            problems.append(f"{it.id}: unknown doc {it.doc_id}")
            continue
        if it.doc_id not in cache:
            cache[it.doc_id] = [_norm(t) for t in page_texts(by_id[it.doc_id])]
        pages = cache[it.doc_id]
        needle = _norm(it.answer)
        for p in it.gold_pages:
            if not (1 <= p <= len(pages)) or needle not in pages[p - 1]:
                problems.append(f"{it.id}: answer {it.answer!r} not found on page {p}")
    return problems
