"""REST API + single-page UI."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import Analyst, extractive_answer, guess_ticker
from .ingest import load_manifest
from .ratios import RATIOS
from .retrieval import load_index
from .tools import Toolbox

STATIC = Path(__file__).parent / "static"
state: dict = {}


def build_toolbox() -> Toolbox:
    return Toolbox(load_index(rerank=os.environ.get("KAP_RERANK", "1") != "0"), load_manifest())


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.setdefault("toolbox", build_toolbox())
    yield


app = FastAPI(title="KAP Analyst", version="0.1.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    ticker: str | None = None
    k: int = Field(5, ge=1, le=10)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    mode: str = Field("auto", pattern="^(auto|agent|extractive)$")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "agent_available": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.get("/api/documents")
def documents() -> list[dict]:
    return [
        {"doc_id": d.doc_id, "ticker": d.ticker, "company": d.company, "title": d.title, "kind": d.kind,
         "url": d.url, "pages": d.pages}
        for d in state["toolbox"].docs
    ]


@app.post("/api/search")
def search(req: SearchRequest) -> list[dict]:
    return state["toolbox"].search_reports(req.query, ticker=req.ticker or guess_ticker(req.query), k=req.k)


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    toolbox: Toolbox = state["toolbox"]
    use_agent = req.mode == "agent" or (req.mode == "auto" and os.environ.get("ANTHROPIC_API_KEY"))
    if req.mode == "agent" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set; use mode=extractive.")
    res = Analyst(toolbox).ask(req.question) if use_agent else extractive_answer(toolbox, req.question)
    return res.__dict__


@app.get("/api/statements/{ticker}/{item}")
def line_item(ticker: str, item: str, period: str = "2024") -> dict:
    try:
        return state["toolbox"].get_line_item(ticker, item, period)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/ratios/{ticker}")
def ratios(ticker: str, period: str = "2024") -> list[dict]:
    try:
        return [state["toolbox"].compute_ratio(ticker, r, period) for r in RATIOS]
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
