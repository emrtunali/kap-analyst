"""Retrieval evaluation: page-level Hit@k, MRR@10 and nDCG@10 across retriever variants."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .ingest import Document, page_texts
from .labels import QAItem, _norm, answer_bearing_pages
from .retrieval import Retriever, unique_pages


@dataclass
class RetrievalReport:
    name: str
    n: int
    hit1: float
    hit3: float
    hit5: float
    hit10: float
    mrr10: float
    ndcg10: float
    ms_per_query: float
    by_lang: dict[str, dict[str, float]] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)


def _ndcg(ranked: list[bool], n_relevant: int, k: int = 10) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, rel in enumerate(ranked[:k]) if rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / idcg if idcg else 0.0


def relevance_sets(items: list[QAItem], docs: list[Document]) -> dict[str, set[tuple[str, int]]]:
    by_id = {d.doc_id: d for d in docs}
    norm_pages = {d.doc_id: [_norm(t) for t in page_texts(by_id[d.doc_id])] for d in docs}
    return {it.id: {(it.doc_id, p) for p in answer_bearing_pages(it, norm_pages[it.doc_id])} for it in items}


def evaluate_retriever(
    name: str,
    retriever: Retriever,
    items: list[QAItem],
    relevant: dict[str, set[tuple[str, int]]],
    ticker_fn: Callable[[str], str | None] | None = None,
) -> RetrievalReport:
    rows = []
    t0 = time.perf_counter()
    for it in items:
        ticker = ticker_fn(it.question) if ticker_fn else None
        pages = unique_pages(retriever.search(it.question, k=50, ticker=ticker))[:10]
        rel = [p in relevant[it.id] for p in pages]
        first = next((i for i, r in enumerate(rel) if r), None)
        rows.append(
            {
                "id": it.id,
                "lang": it.lang,
                "rank": first,
                "ndcg": _ndcg(rel, len(relevant[it.id])),
            }
        )
    elapsed = (time.perf_counter() - t0) * 1000 / max(len(items), 1)

    def agg(rs: list[dict]) -> dict[str, float]:
        def hit(k: int) -> float:
            return statistics.mean(r["rank"] is not None and r["rank"] < k for r in rs)

        return {
            "n": len(rs),
            "hit@1": hit(1),
            "hit@3": hit(3),
            "hit@5": hit(5),
            "hit@10": hit(10),
            "mrr@10": statistics.mean(1 / (r["rank"] + 1) if r["rank"] is not None else 0.0 for r in rs),
            "ndcg@10": statistics.mean(r["ndcg"] for r in rs),
        }

    overall = agg(rows)
    by_lang = {lang: agg([r for r in rows if r["lang"] == lang]) for lang in sorted({r["lang"] for r in rows})}
    return RetrievalReport(
        name=name,
        n=len(rows),
        hit1=overall["hit@1"],
        hit3=overall["hit@3"],
        hit5=overall["hit@5"],
        hit10=overall["hit@10"],
        mrr10=overall["mrr@10"],
        ndcg10=overall["ndcg@10"],
        ms_per_query=elapsed,
        by_lang=by_lang,
        misses=[r["id"] for r in rows if r["rank"] is None or r["rank"] >= 5],
    )


def to_markdown(reports: list[RetrievalReport]) -> str:
    head = (
        "| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | ms/query |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r.name} | {r.hit1:.3f} | {r.hit3:.3f} | {r.hit5:.3f} | {r.hit10:.3f} | {r.mrr10:.3f} | "
        f"{r.ndcg10:.3f} | {r.ms_per_query:.1f} |\n"
        for r in reports
    )
    return head + body


def to_json(reports: list[RetrievalReport]) -> str:
    return json.dumps([r.__dict__ for r in reports], ensure_ascii=False, indent=2)
