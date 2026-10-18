"""Lexical, dense, hybrid and re-ranked retrieval over page-aware chunks.

All retrievers share one interface: ``search(query, k, ticker=None) -> list[Hit]``.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from . import config
from .ingest import Chunk
from .textnorm import analyze


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 10, ticker: str | None = None) -> list[Hit]: ...


def _allowed(chunks: list[Chunk], ticker: str | None) -> np.ndarray:
    if ticker is None:
        return np.ones(len(chunks), dtype=bool)
    return np.array([c.ticker == ticker.upper() for c in chunks])


class BM25:
    """Okapi BM25 with Turkish-aware analysis (``prefix=None`` disables stemming)."""

    def __init__(self, chunks: list[Chunk], prefix: int | None = 5, k1: float = 1.2, b: float = 0.75):
        self.chunks, self.prefix, self.k1, self.b = chunks, prefix, k1, b
        docs = [analyze(c.text, prefix) for c in chunks]
        self.doc_len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(docs) else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.chunks))
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in set(analyze(query, self.prefix)):
            for i, tf in self.postings.get(term, ()):
                s[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[i])
        return s

    def search(self, query: str, k: int = 10, ticker: str | None = None) -> list[Hit]:
        s = self.scores(query)
        s[~_allowed(self.chunks, ticker)] = -np.inf
        top = np.argsort(-s)[:k]
        return [Hit(self.chunks[i], float(s[i])) for i in top if np.isfinite(s[i]) and s[i] > 0]


class Dense:
    """Bi-encoder retrieval. E5 models need ``query:`` / ``passage:`` prefixes; BGE-M3 needs none."""

    def __init__(self, chunks: list[Chunk], model_name: str = config.EMBED_MODEL, cache: Path | None = None):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        e5 = "e5" in model_name.lower()
        self.q_prefix, self.p_prefix = ("query: ", "passage: ") if e5 else ("", "")
        if cache and cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(
                [f"{self.p_prefix}{c.text}" for c in chunks], batch_size=32, normalize_embeddings=True,
                show_progress_bar=True,
            ).astype(np.float32)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.emb)
        if len(self.emb) != len(chunks):
            raise ValueError("Embedding cache does not match chunks; delete it and rebuild the index.")

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([f"{self.q_prefix}{query}"], normalize_embeddings=True)[0]
        return self.emb @ q

    def search(self, query: str, k: int = 10, ticker: str | None = None) -> list[Hit]:
        s = self.scores(query)
        s[~_allowed(self.chunks, ticker)] = -np.inf
        top = np.argsort(-s)[:k]
        return [Hit(self.chunks[i], float(s[i])) for i in top if np.isfinite(s[i])]


class Hybrid:
    """Reciprocal Rank Fusion (Cormack et al., 2009) of lexical and dense rankings."""

    def __init__(self, lexical: BM25, dense: Dense, rrf_k: int = 60, depth: int = 100):
        self.lexical, self.dense, self.rrf_k, self.depth = lexical, dense, rrf_k, depth
        self.chunks = lexical.chunks

    def search(self, query: str, k: int = 10, ticker: str | None = None) -> list[Hit]:
        fused: dict[str, float] = defaultdict(float)
        by_id: dict[str, Chunk] = {}
        for ranking in (self.lexical.search(query, self.depth, ticker), self.dense.search(query, self.depth, ticker)):
            for rank, hit in enumerate(ranking):
                fused[hit.chunk.chunk_id] += 1.0 / (self.rrf_k + rank + 1)
                by_id[hit.chunk.chunk_id] = hit.chunk
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(by_id[cid], fused[cid]) for cid in best]


class Reranked:
    """Cross-encoder re-ranking of the first-stage candidates."""

    def __init__(self, first_stage: Retriever, model_name: str = config.RERANK_MODEL, candidates: int = 30):
        from sentence_transformers import CrossEncoder

        self.first_stage, self.candidates = first_stage, candidates
        self.model = CrossEncoder(model_name, max_length=384)

    def search(self, query: str, k: int = 10, ticker: str | None = None) -> list[Hit]:
        cands = self.first_stage.search(query, self.candidates, ticker)
        if not cands:
            return []
        scores = self.model.predict([(query, h.chunk.text) for h in cands], batch_size=32)
        order = np.argsort(-np.asarray(scores))[:k]
        return [Hit(cands[i].chunk, float(scores[i])) for i in order]


def unique_pages(hits: list[Hit]) -> list[tuple[str, int]]:
    """Collapse a chunk ranking into a ranking of distinct (doc_id, page) pairs."""
    seen, out = set(), []
    for h in hits:
        key = (h.chunk.doc_id, h.chunk.page)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def embedding_cache(model_name: str, index_dir: Path = config.INDEX_DIR) -> Path:
    return index_dir / f"embeddings-{model_name.split('/')[-1]}.npy"


def load_index(index_dir: Path = config.INDEX_DIR, rerank: bool = True) -> Retriever:
    """Build the production retriever (hybrid + optional cross-encoder) from a saved index."""
    from .ingest import load_chunks

    chunks = load_chunks(index_dir / "chunks.jsonl")
    meta = json.loads((index_dir / "meta.json").read_text())
    hybrid = Hybrid(BM25(chunks), Dense(chunks, meta["embed_model"], embedding_cache(meta["embed_model"], index_dir)))
    return Reranked(hybrid) if rerank else hybrid
