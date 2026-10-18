"""Parsing of numbers as printed in Turkish financial statements.

Turkish reports use "." as the thousands separator and "," as the decimal mark, and
show negatives in parentheses, often with a stray space: ``( 602.522)``. A dash means
zero / not applicable.
"""

from __future__ import annotations

import re

_NUM_RE = re.compile(r"^\(?\s*-?\s*\d{1,3}(?:\.\d{3})*(?:,\d+)?\s*\)?$|^\(?\s*-?\s*\d+(?:,\d+)?\s*\)?$")


def is_number_token(token: str) -> bool:
    token = token.strip()
    return token in {"-", "–"} or bool(_NUM_RE.match(token))


def parse_tr_number(token: str) -> float | None:
    """Parse ``"1.399.606"`` → 1399606.0, ``"( 853)"`` → -853.0, ``"82,15"`` → 82.15, ``"-"`` → 0.0."""
    raw = token.strip()
    if raw in {"-", "–"}:
        return 0.0
    if not is_number_token(raw):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    body = raw.strip("()").strip()
    if body.startswith("-"):
        negative = True
        body = body[1:].strip()
    body = body.replace(".", "").replace(",", ".")
    value = float(body)
    return -value if negative else value


def format_tr(value: float, decimals: int = 0) -> str:
    """Format a number the Turkish way (thousands '.', decimal ',')."""
    s = f"{abs(value):,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"-{s}" if value < 0 else s
