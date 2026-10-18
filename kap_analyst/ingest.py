"""Download, verify and parse source PDFs into page-aware chunks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import pymupdf

from . import config
from .textnorm import squash_ws


@dataclass(frozen=True)
class Document:
    doc_id: str
    ticker: str
    company: str
    title: str
    kind: str
    period: str
    currency_unit: str | None
    url: str
    sha256: str
    pages: int

    @property
    def path(self) -> Path:
        return config.RAW_DIR / f"{self.doc_id}.pdf"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ticker: str
    page: int  # 1-based PDF page index
    text: str


def load_manifest(path: Path = config.MANIFEST) -> list[Document]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Document(**d) for d in data["documents"]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(docs: list[Document], force: bool = False) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "kap-analyst/0.1 (research; +https://github.com/emrtunali/kap-analyst)"}
    with httpx.Client(follow_redirects=True, timeout=120, headers=headers) as client:
        for doc in docs:
            if doc.path.exists() and not force and sha256_file(doc.path) == doc.sha256:
                print(f"✓ {doc.doc_id} (cached)")
                continue
            resp = client.get(doc.url)
            resp.raise_for_status()
            doc.path.write_bytes(resp.content)
            digest = sha256_file(doc.path)
            if digest != doc.sha256:
                raise RuntimeError(
                    f"{doc.doc_id}: checksum mismatch ({digest}). The source file changed, so the "
                    "page-level eval labels may no longer be valid."
                )
            print(f"✓ {doc.doc_id} downloaded ({len(resp.content) / 1e6:.1f} MB)")


def page_texts(doc: Document) -> list[str]:
    """Plain text per page in reading order (index 0 = page 1)."""
    with pymupdf.open(doc.path) as pdf:
        return [page.get_text("text") for page in pdf]


def chunk_page(text: str, words: int = config.CHUNK_WORDS, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    toks = squash_ws(text).split(" ")
    if len(toks) < 8:
        return []
    out, step = [], max(1, words - overlap)
    for start in range(0, len(toks), step):
        window = toks[start : start + words]
        out.append(" ".join(window))
        if start + words >= len(toks):
            break
    return out


def build_chunks(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for page_no, text in enumerate(page_texts(doc), start=1):
            for i, piece in enumerate(chunk_page(text)):
                chunks.append(Chunk(f"{doc.doc_id}:p{page_no}:c{i}", doc.doc_id, doc.ticker, page_no, piece))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
