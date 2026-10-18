"""Agent loop and API tests with a scripted fake Claude client (no network, no API key)."""

from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from kap_analyst.agent import Analyst, extractive_answer, guess_ticker
from kap_analyst.tools import Toolbox


class FakeMessages:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.script.pop(0)


def fake_client(script):
    return NS(beta=NS(messages=FakeMessages(script)))


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def response(content, stop):
    return NS(content=content, stop_reason=stop, model="claude-opus-5-5", usage=NS(input_tokens=10, output_tokens=5))


@pytest.fixture
def toolbox(bm25):
    return Toolbox(retriever=bm25, docs=[])


def test_agent_runs_tool_loop_and_collects_citations(toolbox):
    client = fake_client([
        response([tool_use("t1", "search_reports", {"query": "yeni mağaza", "ticker": "BIMAS"})], "tool_use"),
        response([text("BİM 2024'te 1.101 yeni mağaza açtı [BIMAS s.19].")], "end_turn"),
    ])
    res = Analyst(toolbox, client=client).ask("BİM kaç mağaza açtı?")
    assert "1.101" in res.answer
    assert {"doc_id": "BIMAS_DOC", "ticker": "BIMAS", "page": 19} in res.citations
    second = client.beta.messages.calls[1]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result"
    assert second[-1]["content"][0]["tool_use_id"] == "t1"
    assert res.usage == {"input_tokens": 20, "output_tokens": 10}


def test_agent_reports_tool_errors_back_to_model(toolbox):
    client = fake_client([
        response([tool_use("t1", "get_line_item", {"ticker": "BIMAS", "item": "revenue", "period": "2024"})],
                 "tool_use"),
        response([text("Bu veri mevcut değil.")], "end_turn"),
    ])
    res = Analyst(toolbox, client=client).ask("BİM hasılatı?")
    sent = client.beta.messages.calls[1]["messages"][-1]["content"][0]
    assert sent["is_error"] is True and "No audited financial statements" in sent["content"]
    assert res.tool_calls[0]["is_error"] is True


def test_agent_handles_refusal(toolbox):
    client = fake_client([response([], "refusal")])
    res = Analyst(toolbox, client=client).ask("x")
    assert res.stop_reason == "refusal" and "declined" in res.answer


def test_agent_requests_use_fallbacks_and_effort(toolbox):
    client = fake_client([response([text("ok")], "end_turn")])
    Analyst(toolbox, client=client).ask("x")
    call = client.beta.messages.calls[0]
    assert call["fallbacks"] == "default" and call["output_config"] == {"effort": "medium"}


@pytest.mark.parametrize(
    ("q", "ticker"),
    [("THY'nin net kârı?", "THYAO"), ("BİM'in mağaza sayısı", "BIMAS"), ("ASELSAN backlog", "ASELS"),
     ("bilim ve teknoloji", None)],
)
def test_guess_ticker(q, ticker):
    assert guess_ticker(q) == ticker


def test_extractive_mode_cites_pages(toolbox):
    res = extractive_answer(toolbox, "THY akaryakıt gideri")
    assert res.citations[0]["page"] == 72 and "[THYAO s.72]" in res.answer


def test_api_search_and_ask(toolbox, monkeypatch):
    from kap_analyst import api

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    api.state["toolbox"] = toolbox
    client = TestClient(api.app)
    assert client.get("/api/health").json() == {"status": "ok", "agent_available": False}
    hits = client.post("/api/search", json={"query": "gıda atığı oranı", "k": 2}).json()
    assert hits[0]["page"] == 58
    ans = client.post("/api/ask", json={"question": "ASELSAN sözleşmeli proje tutarı"}).json()
    assert ans["stop_reason"] == "extractive" and ans["citations"][0]["ticker"] == "ASELS"
    assert client.post("/api/ask", json={"question": "x y", "mode": "agent"}).status_code == 503
    assert "<title>KAP Analyst</title>" in client.get("/").text
