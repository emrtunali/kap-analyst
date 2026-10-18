"""Financial ratios computed from extracted line items (never from model arithmetic)."""

from __future__ import annotations

from collections.abc import Callable

from .statements import StatementSet, get_line_item

PREV = {"2024": "2023", "2025": "2024", "2023": "2022"}


def _avg(stmts: StatementSet, item: str, period: str, used: list[dict]) -> float:
    cur = get_line_item(stmts, item, period)
    prev = get_line_item(stmts, item, PREV[period])
    used += [cur, prev]
    return (cur["value"] + prev["value"]) / 2


def _v(stmts: StatementSet, item: str, period: str, used: list[dict]) -> float:
    li = get_line_item(stmts, item, period)
    used.append(li)
    return li["value"]


RATIOS: dict[str, tuple[str, Callable[[StatementSet, str, list[dict]], float]]] = {
    "gross_margin": ("Brüt kâr / Hasılat", lambda s, p, u: _v(s, "gross_profit", p, u) / _v(s, "revenue", p, u)),
    "operating_margin": (
        "Esas faaliyet kârı / Hasılat",
        lambda s, p, u: _v(s, "operating_profit", p, u) / _v(s, "revenue", p, u),
    ),
    "net_margin": ("Dönem kârı / Hasılat", lambda s, p, u: _v(s, "net_profit", p, u) / _v(s, "revenue", p, u)),
    "roe": (
        "Ana ortaklık payı net kâr / Ortalama ana ortaklık özkaynakları",
        lambda s, p, u: _v(s, "net_profit_parent", p, u) / _avg(s, "equity_parent", p, u),
    ),
    "roa": (
        "Dönem kârı / Ortalama toplam varlıklar",
        lambda s, p, u: _v(s, "net_profit", p, u) / _avg(s, "total_assets", p, u),
    ),
    "current_ratio": (
        "Dönen varlıklar / Kısa vadeli yükümlülükler",
        lambda s, p, u: _v(s, "current_assets", p, u) / _v(s, "current_liabilities", p, u),
    ),
    "liabilities_to_equity": (
        "(Kısa + uzun vadeli yükümlülükler) / Toplam özkaynaklar",
        lambda s, p, u: (_v(s, "current_liabilities", p, u) + _v(s, "non_current_liabilities", p, u))
        / _v(s, "total_equity", p, u),
    ),
    "revenue_growth": (
        "Hasılat (t) / Hasılat (t-1) − 1",
        lambda s, p, u: _v(s, "revenue", p, u) / _v(s, "revenue", PREV[p], u) - 1,
    ),
    "capex_to_revenue": (
        "|Duran varlık alımları| / Hasılat",
        lambda s, p, u: abs(_v(s, "capex", p, u)) / _v(s, "revenue", p, u),
    ),
    "cash_conversion": (
        "İşletme faaliyetlerinden net nakit / Dönem kârı",
        lambda s, p, u: _v(s, "operating_cash_flow", p, u) / _v(s, "net_profit", p, u),
    ),
}


def compute_ratio(stmts: StatementSet, ratio: str, period: str = "2024") -> dict:
    if ratio not in RATIOS:
        raise KeyError(f"Unknown ratio '{ratio}'. Available: {', '.join(RATIOS)}")
    formula, fn = RATIOS[ratio]
    used: list[dict] = []
    value = fn(stmts, period, used)
    return {
        "ratio": ratio,
        "period": period,
        "value": round(value, 6),
        "formula": formula,
        "inputs": [{k: li[k] for k in ("item", "period", "value", "page")} for li in used],
    }
