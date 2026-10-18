"""Tools exposed to the agent. Each returns JSON-serialisable data with page citations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .ingest import Document
from .ratios import RATIOS, compute_ratio
from .retrieval import Retriever
from .statements import CANONICAL_ITEMS, StatementSet, extract_statements, get_line_item

TOOL_SPECS = [
    {
        "name": "search_reports",
        "description": (
            "Search the indexed KAP disclosures (annual reports, audited financial statements) and return "
            "the most relevant passages with document id and PDF page number. Use for any qualitative "
            "or numeric question. Queries work in Turkish or English; Turkish keywords match best."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, ideally in Turkish."},
                "ticker": {
                    "type": "string",
                    "description": "Optional BIST ticker to restrict the search, e.g. THYAO, BIMAS, ASELS.",
                },
                "k": {"type": "integer", "description": "Number of passages (1-10).", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_line_item",
        "description": (
            "Read one figure from a company's audited financial statements (balance sheet, income "
            "statement, cash flow). Values are in million TL and come with the source page. "
            f"Canonical items: {', '.join(CANONICAL_ITEMS)}. A literal Turkish row label also works."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "item": {"type": "string"},
                "period": {"type": "string", "description": "Fiscal year, e.g. '2024' or '2023'."},
            },
            "required": ["ticker", "item", "period"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compute_ratio",
        "description": (
            "Compute a financial ratio deterministically from the statements and return its formula and "
            f"cited inputs. Always use this instead of doing arithmetic yourself. Ratios: {', '.join(RATIOS)}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "ratio": {"type": "string", "enum": list(RATIOS)},
                "period": {"type": "string"},
            },
            "required": ["ticker", "ratio", "period"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class Toolbox:
    retriever: Retriever
    docs: list[Document]
    _statements: dict[str, StatementSet] = field(default_factory=dict)

    def statements_for(self, ticker: str) -> StatementSet:
        ticker = ticker.upper()
        doc = next((d for d in self.docs if d.ticker == ticker and d.kind == "financial_statements"), None)
        if doc is None:
            have = sorted({d.ticker for d in self.docs if d.kind == "financial_statements"})
            raise KeyError(f"No audited financial statements indexed for {ticker}. Available: {have}")
        if doc.doc_id not in self._statements:
            self._statements[doc.doc_id] = extract_statements(doc)
        return self._statements[doc.doc_id]

    def search_reports(self, query: str, ticker: str | None = None, k: int = 5) -> list[dict]:
        hits = self.retriever.search(query, k=max(1, min(k, 10)), ticker=ticker)
        return [
            {"doc_id": h.chunk.doc_id, "ticker": h.chunk.ticker, "page": h.chunk.page, "text": h.chunk.text}
            for h in hits
        ]

    def get_line_item(self, ticker: str, item: str, period: str) -> dict:
        return get_line_item(self.statements_for(ticker), item, period)

    def compute_ratio(self, ticker: str, ratio: str, period: str) -> dict:
        return compute_ratio(self.statements_for(ticker), ratio, period)

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Execute a tool call; returns (json_result, is_error)."""
        fn = {"search_reports": self.search_reports, "get_line_item": self.get_line_item,
              "compute_ratio": self.compute_ratio}.get(name)
        if fn is None:
            return f"Unknown tool: {name}", True
        try:
            return json.dumps(fn(**args), ensure_ascii=False), False
        except (KeyError, TypeError, ValueError) as exc:
            return f"Error: {exc}", True
