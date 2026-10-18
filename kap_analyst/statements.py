"""Deterministic line-item extraction from Turkish financial statements.

LLMs are unreliable at reading numbers out of dense PDF tables, so figures used for
ratio calculations never come from free-text generation. Instead we rebuild table rows
from PyMuPDF word coordinates:

1. find the statement pages by their titles,
2. locate the period columns from the right edge of the "2024"/"2023" header tokens,
3. group words into visual rows, split label / note reference / values by x-position,
4. stitch labels that wrap onto a second line.

Every extracted value keeps its page number so the agent can cite it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

from .ingest import Document
from .numbers import is_number_token, parse_tr_number
from .textnorm import fold

STATEMENT_TITLES = {
    "balance_sheet": "finansal durum tablosu",
    "income_statement": "kar veya zarar",
    "cash_flow": "nakit akis tablosu",
}

# Canonical item → candidate row labels, in priority order (matched after `fold`).
CANONICAL_ITEMS: dict[str, tuple[str, list[str]]] = {
    "revenue": ("income_statement", ["Hasılat"]),
    "cost_of_sales": ("income_statement", ["Satışların Maliyeti (-)", "Satışların Maliyeti"]),
    "gross_profit": ("income_statement", ["BRÜT KAR", "Brüt Kâr"]),
    "general_admin_expenses": ("income_statement", ["Genel Yönetim Giderleri (-)"]),
    "marketing_expenses": ("income_statement", ["Pazarlama Giderleri (-)"]),
    "operating_profit": ("income_statement", ["ESAS FAALİYET KARI", "ESAS FAALİYET KÂRI"]),
    "finance_income": ("income_statement", ["Finansman Gelirleri"]),
    "finance_expenses": ("income_statement", ["Finansman Giderleri (-)"]),
    "pretax_profit": ("income_statement", ["SÜRDÜRÜLEN FAALİYETLER VERGİ ÖNCESİ KARI"]),
    "net_profit": ("income_statement", ["SÜRDÜRÜLEN FAALİYETLER DÖNEM KARI", "DÖNEM KARI"]),
    "net_profit_parent": ("income_statement", ["- Ana Ortaklık Payları", "Ana Ortaklık Payları"]),
    "eps": ("income_statement", ["Pay Başına Kar (Kr)"]),
    "total_comprehensive_income": ("income_statement", ["TOPLAM KAPSAMLI GELİR"]),
    "cash": ("balance_sheet", ["Nakit ve Nakit Benzerleri"]),
    "inventories": ("balance_sheet", ["Stoklar"]),
    "current_assets": ("balance_sheet", ["TOPLAM DÖNEN VARLIKLAR"]),
    "property_plant_equipment": ("balance_sheet", ["Maddi Duran Varlıklar"]),
    "right_of_use_assets": ("balance_sheet", ["Kullanım Hakkı Varlıkları"]),
    "non_current_assets": ("balance_sheet", ["TOPLAM DURAN VARLIKLAR"]),
    "total_assets": ("balance_sheet", ["TOPLAM VARLIKLAR"]),
    "current_liabilities": ("balance_sheet", ["TOPLAM KISA VADELİ YÜKÜMLÜLÜKLER"]),
    "non_current_liabilities": ("balance_sheet", ["TOPLAM UZUN VADELİ YÜKÜMLÜLÜKLER"]),
    "paid_in_capital": ("balance_sheet", ["Ödenmiş Sermaye"]),
    "retained_earnings": ("balance_sheet", ["Geçmiş Yıllar Karları"]),
    "equity_parent": ("balance_sheet", ["Ana ortaklığa ait özkaynaklar"]),
    "total_equity": ("balance_sheet", ["TOPLAM ÖZKAYNAKLAR"]),
    "total_liabilities_and_equity": ("balance_sheet", ["TOPLAM KAYNAKLAR"]),
    "operating_cash_flow": ("cash_flow", ["İşletme Faaliyetlerinden Kaynaklanan Net Nakit"]),
    "investing_cash_flow": ("cash_flow", ["Yatırım Faaliyetlerinden Kaynaklanan Net Nakit Akışları"]),
    "financing_cash_flow": ("cash_flow", ["Finansman Faaliyetlerinde Kullanılan Net Nakit"]),
    "capex": (
        "cash_flow",
        ["Maddi ve Maddi Olmayan Duran Varlık Alımından Kaynaklanan Nakit Çıkışları"],
    ),
    "depreciation_amortization": ("cash_flow", ["Amortisman ve İtfa Gideri İle İlgili Düzeltmeler"]),
    "dividends_received": ("cash_flow", ["Alınan Temettüler"]),
    "cash_end_of_period": ("cash_flow", ["Dönem Sonu Nakit ve Nakit Benzerleri"]),
}


@dataclass
class Row:
    statement: str
    page: int
    label: str
    values: dict[str, float]
    note: str | None = None
    continued_from: str | None = None  # label fragment from the previous wrapped line

    @property
    def full_label(self) -> str:
        return f"{self.continued_from} {self.label}".strip() if self.continued_from else self.label


@dataclass
class StatementSet:
    doc_id: str
    rows: list[Row] = field(default_factory=list)

    def find(self, label: str, statement: str | None = None) -> list[Row]:
        target = _key(label)
        pool = [r for r in self.rows if statement is None or r.statement == statement]
        exact = [r for r in pool if _key(r.label) == target or _key(r.full_label) == target]
        if exact:
            return exact
        # Fallback: all target tokens present in the (stitched) label.
        toks = target.split()
        return [r for r in pool if all(t in _key(r.full_label).split() for t in toks)]


def _key(label: str) -> str:
    return " ".join(fold(label).replace("(", " ").replace(")", " ").replace("-", " ").split())


def _merge_paren_tokens(words: list[tuple]) -> list[tuple]:
    """Turn ['(', '602.522)'] into ['( 602.522)'] so negatives parse as one value."""
    out: list[tuple] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w[4] == "(" and i + 1 < len(words) and words[i + 1][4].endswith(")"):
            nxt = words[i + 1]
            out.append((w[0], w[1], nxt[2], nxt[3], f"({nxt[4]}"))
            i += 2
            continue
        out.append(w)
        i += 1
    return out


def _same_line(a: tuple, b: tuple, min_overlap: float = 0.5) -> bool:
    """Bold totals sit a few points lower than their labels, so compare vertical overlap, not centres."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap >= min_overlap * min(a[3] - a[1], b[3] - b[1])


def _rows_of(page: pymupdf.Page) -> list[list[tuple]]:
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list[tuple]] = []
    for w in words:
        if rows and _same_line(rows[-1][0], w):
            rows[-1].append(w)
        else:
            rows.append([w])
    return [_merge_paren_tokens(sorted(r, key=lambda w: w[0])) for r in rows]


def _period_columns(rows: list[list[tuple]], page_width: float) -> tuple[dict[str, float], float]:
    """Right-edge x of each period column (from header tokens like '31 Aralık 2024') and the header's bottom y."""
    cols: dict[str, float] = {}
    header_bottom = 0.0
    for row in rows[:25]:
        for w in row:
            if w[4] in {"2024", "2023", "2025", "2022"} and w[0] > page_width * 0.5:
                cols.setdefault(w[4], w[2])
                header_bottom = max(header_bottom, w[3])
    return cols, header_bottom


def classify_page(page: pymupdf.Page) -> str | None:
    head = fold(" ".join(page.get_text("text").split()[:40]))
    if "dipnot" in head and "iliskin dipnotlar" in head:
        return None
    for name, title in STATEMENT_TITLES.items():
        if title in head:
            return name
    return None


def extract_statements(doc: Document) -> StatementSet:
    out = StatementSet(doc.doc_id)
    with pymupdf.open(doc.path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            statement = classify_page(page)
            if statement is None or statement == "cash_flow" and "ozkaynak" in fold(page.get_text("text")[:300]):
                continue
            rows = _rows_of(page)
            cols, header_bottom = _period_columns(rows, page.rect.width)
            if len(cols) < 2:
                continue
            pending: str | None = None
            for row in rows:
                if row[0][3] <= header_bottom + 1:
                    continue  # title block and column headers
                values: dict[str, float] = {}
                label_words, note = [], None
                for w in row:
                    text = w[4]
                    near = min(cols.items(), key=lambda kv: abs(kv[1] - w[2]))
                    if is_number_token(text) and abs(near[1] - w[2]) <= 22:
                        values[near[0]] = parse_tr_number(text)
                    elif is_number_token(text) or (any(c.isdigit() for c in text) and "-" in text and w[0] > 250):
                        note = text  # note reference column, e.g. "25" or "7-17"
                    else:
                        label_words.append(text)
                label = " ".join(label_words).strip()
                if values:
                    if label or pending:
                        out.rows.append(Row(statement, page_no, label or pending or "", values, note,
                                            continued_from=pending if label else None))
                    pending = None
                elif label and not label.isupper() and len(label) > 3:
                    # A label-only line is either a section header or the first half of a
                    # wrapped label; keep it so the next value row can be stitched to it.
                    pending = label
                else:
                    pending = None
    return out


def get_line_item(stmts: StatementSet, item: str, period: str = "2024") -> dict:
    """Look up a canonical item (e.g. ``revenue``) or a literal row label."""
    statement, candidates = CANONICAL_ITEMS.get(item, (None, [item]))
    for cand in candidates:
        rows = [r for r in stmts.find(cand, statement) if period in r.values]
        if rows:
            r = rows[0]
            return {
                "item": item,
                "label": r.full_label,
                "period": period,
                "value": r.values[period],
                "unit": "milyon TL",
                "page": r.page,
                "statement": r.statement,
                "ambiguous": len(rows) > 1,
            }
    raise KeyError(f"'{item}' not found for period {period} in {stmts.doc_id}")
