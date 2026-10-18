"""Tool-using analyst agent on the Claude API (manual agentic loop).

The model decides when to search the reports, when to read a statement line item and when
to compute a ratio; every number in the final answer must come from a tool result and be
cited as [TICKER s.<page>].
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from . import config
from .textnorm import tr_lower
from .tools import TOOL_SPECS, Toolbox

SYSTEM_PROMPT = """You are a financial analyst assistant for companies listed on Borsa İstanbul.
You answer questions using only the indexed KAP disclosures, which you reach through tools.

How to work:
- Use search_reports to find passages. Restrict by ticker when the company is clear.
- For figures from audited statements (revenue, profit, assets, cash flows) prefer get_line_item.
- For any ratio, margin or growth rate call compute_ratio. Do not do arithmetic yourself.
- If the tools do not contain the answer, say so plainly. Never guess a number.

Answer format:
- Reply in the language of the question.
- Cite every figure and claim as [TICKER s.<page>], using the page numbers returned by the tools.
- Amounts from the THYAO statements are in million TL; say so.
- Keep it short: the answer first, then one or two sentences of context."""

TICKER_HINTS = {
    "THYAO": ["thy", "türk hava", "turkish airlines", "thyao"],
    "BIMAS": ["bim", "bimas"],
    "ASELS": ["aselsan", "asels"],
}


@dataclass
class AgentResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


def _citations_from(name: str, result: Any) -> list[dict]:
    if name == "search_reports":
        return [{"doc_id": r["doc_id"], "ticker": r["ticker"], "page": r["page"]} for r in result]
    if name == "get_line_item":
        return [{"item": result["item"], "page": result["page"]}]
    if name == "compute_ratio":
        return [{"item": i["item"], "page": i["page"]} for i in result["inputs"]]
    return []


class Analyst:
    def __init__(self, toolbox: Toolbox, client: Any | None = None, model: str = config.LLM_MODEL,
                 max_turns: int = 8, effort: str = "medium"):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.toolbox, self.client, self.model = toolbox, client, model
        self.max_turns, self.effort = max_turns, effort

    def ask(self, question: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": question}]
        result = AgentResult(answer="", model=self.model)
        for _ in range(self.max_turns):
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=TOOL_SPECS,
                messages=messages,
                output_config={"effort": self.effort},
                # On a safety decline the API re-runs the request on a fallback model in the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            result.stop_reason = response.stop_reason
            result.model = getattr(response, "model", self.model)
            usage = getattr(response, "usage", None)
            for key in ("input_tokens", "output_tokens"):
                result.usage[key] = result.usage.get(key, 0) + int(getattr(usage, key, 0) or 0)

            if response.stop_reason == "refusal":
                result.answer = "The request was declined by the model's safety system."
                return result

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                result.answer = "".join(b.text for b in response.content if b.type == "text").strip()
                return result

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_uses:
                output, is_error = self.toolbox.call(block.name, dict(block.input))
                result.tool_calls.append({"name": block.name, "input": dict(block.input), "is_error": is_error})
                if not is_error:
                    result.citations += _citations_from(block.name, json.loads(output))
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error}
                )
            messages.append({"role": "user", "content": tool_results})
        result.answer = "Stopped after the maximum number of tool turns without a final answer."
        return result


def guess_ticker(question: str) -> str | None:
    q = tr_lower(question)
    for ticker, hints in TICKER_HINTS.items():
        if any(re.search(rf"\b{re.escape(h)}\b", q) for h in hints):
            return ticker
    return None


def extractive_answer(toolbox: Toolbox, question: str, k: int = 3) -> AgentResult:
    """No-LLM fallback: return the top cited passages so the app still works without an API key."""
    hits = toolbox.search_reports(question, ticker=guess_ticker(question), k=k)
    lines = [f"[{h['ticker']} s.{h['page']}] {h['text'][:400]}…" for h in hits]
    return AgentResult(
        answer="\n\n".join(lines) or "No matching passage found.",
        citations=[{"doc_id": h["doc_id"], "ticker": h["ticker"], "page": h["page"]} for h in hits],
        stop_reason="extractive",
        model=None,
    )
