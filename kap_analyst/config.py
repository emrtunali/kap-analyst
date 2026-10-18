from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("KAP_ANALYST_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = Path(os.environ.get("KAP_ANALYST_INDEX", DATA_DIR / "index"))
MANIFEST = DATA_DIR / "manifest.json"
EVAL_DIR = DATA_DIR / "eval"
RESULTS_DIR = ROOT / "results"

EMBED_MODEL = os.environ.get("KAP_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("KAP_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
BASELINE_EMBED_MODEL = "intfloat/multilingual-e5-small"
QUALITY_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = os.environ.get("KAP_LLM_MODEL", "claude-opus-5-5")

CHUNK_WORDS = 160
CHUNK_OVERLAP = 40
