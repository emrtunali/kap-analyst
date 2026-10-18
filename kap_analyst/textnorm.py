"""Turkish-aware text normalisation for lexical retrieval.

Python's ``str.lower()`` maps "I" to "i" and "İ" to "i̇" (i + combining dot), which
silently breaks matching for Turkish text ("KARI" should become "karı", not "kari").
We lowercase with Turkish rules and fold the dotted/dotless pairs consistently.

For stemming we use prefix truncation (the "F5" stemmer): Turkish is agglutinative,
so ``kârlılığı``, ``kârlılık`` and ``kârlılığın`` share their first five letters.
Can et al. (2008, "Information retrieval on Turkish texts", JASIST) showed fixed-prefix truncation is competitive with
morphological analysers for Turkish IR, and it needs no external dependency.
"""

from __future__ import annotations

import re
import unicodedata

_TR_UPPER_MAP = str.maketrans({"I": "ı", "İ": "i"})
_TOKEN_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+", re.UNICODE)

# Small closed-class stopword list (function words that carry no retrieval signal).
STOPWORDS = frozenset(
    """
    ve ile bir bu şu o da de mi mı mu mü için gibi olarak olan olan ise ama fakat veya ya
    çok daha en her hem ki ne nasıl kaç hangi neden niçin nedir midir mıdır dır dir tır tir
    the of and to in for on by is are what which how many much
    """.split()
)


def tr_lower(text: str) -> str:
    """Lowercase using Turkish casing rules (I→ı, İ→i)."""
    return unicodedata.normalize("NFC", text.translate(_TR_UPPER_MAP).lower())


def fold(text: str) -> str:
    """Aggressive ASCII-ish fold used for fuzzy label matching ("Kâr" == "kar", "İ" == "i")."""
    text = tr_lower(text)
    text = text.replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(tr_lower(text))


def stem(token: str, prefix: int = 5) -> str:
    if token[0].isdigit():
        return token
    return fold(token)[:prefix]


def analyze(text: str, prefix: int | None = 5) -> list[str]:
    """Tokenise → drop stopwords → optional F-prefix stemming."""
    toks = [t for t in tokenize(text) if t not in STOPWORDS]
    if prefix is None:
        return toks
    return [stem(t, prefix) for t in toks]


def squash_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
