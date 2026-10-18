import pytest

from kap_analyst.numbers import format_tr, parse_tr_number
from kap_analyst.textnorm import analyze, fold, tr_lower


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.399.606", 1399606.0),
        ("745.430", 745430.0),
        ("( 602.522)", -602522.0),
        ("(8)", -8.0),
        ("82,15", 82.15),
        ("-", 0.0),
        ("78", 78.0),
    ],
)
def test_parse_tr_number(raw, expected):
    assert parse_tr_number(raw) == expected


@pytest.mark.parametrize("raw", ["Hasılat", "7-17", "2024a", ""])
def test_parse_tr_number_rejects_text(raw):
    assert parse_tr_number(raw) is None


def test_format_tr_roundtrip():
    assert format_tr(1399606) == "1.399.606"
    assert format_tr(-82.15, 2) == "-82,15"


def test_turkish_lowercase_handles_dotted_and_dotless_i():
    assert tr_lower("KARI") == "karı"
    assert tr_lower("İŞLETME") == "işletme"
    assert fold("BRÜT KÂR") == fold("brut kar")


def test_f5_stemming_conflates_inflections():
    forms = ["kârlılığı", "kârlılık", "kârlılığın"]
    stems = {analyze(f)[0] for f in forms}
    assert len(stems) == 1


def test_stopwords_removed():
    assert analyze("THY'nin net kârı ne kadar?", prefix=None) == ["thy", "nin", "net", "kârı", "kadar"]
