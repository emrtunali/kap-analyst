from pathlib import Path

import pymupdf
import pytest

from kap_analyst.ingest import Chunk, Document
from kap_analyst.retrieval import BM25


@pytest.fixture
def chunks() -> list[Chunk]:
    texts = [
        ("THYAO", 11, "Hasılat 745.430 Satışların Maliyeti BRÜT KAR 142.908 net dönem kârı"),
        ("THYAO", 72, "Akaryakıt giderleri 201.778 milyon TL, personel giderleri 128.207"),
        ("BIMAS", 19, "2024 yılında 1.101 yeni mağaza açılışı ile organik büyümesine devam eden BİM"),
        ("BIMAS", 58, "Oluşan gıda atığı 50.813 ton, gıda atık miktarının satışlara oranı %0,56"),
        ("ASELS", 12, "sözleşmeli proje tutarı 31 Aralık 2024 tarihi itibarıyla 13,9 Milyar ABD Doları"),
    ]
    return [Chunk(f"{t}:p{p}:c0", f"{t}_DOC", t, p, txt) for t, p, txt in texts]


@pytest.fixture
def bm25(chunks) -> BM25:
    return BM25(chunks)


def _write_statement_pdf(path: Path) -> None:
    """A miniature income statement laid out like a KAP filing (labels left, notes middle, values right)."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    font = pymupdf.Font("cjk")  # bundled with PyMuPDF; unlike Helvetica it has ı, ş, ğ, İ glyphs
    page.insert_font(fontname="tr", fontbuffer=font.buffer)

    def put(pos, text, fontsize):
        page.insert_text(pos, text, fontsize=fontsize, fontname="tr")

    put((57, 50), "Konsolide Kar veya Zarar ve Diğer Kapsamlı Gelir Tablosu", 9)
    put((300, 120), "Dipnot", 8)
    put((410, 120), "2024", 8)
    put((505, 120), "2023", 8)
    rows = [
        (150, "Hasılat", "25", "745.430", "504.398"),
        (165, "Satışların Maliyeti (-)", "26", "( 602.522)", "( 384.952)"),
        (180, "BRÜT KAR", "", "142.908", "119.446"),
    ]

    def right(x_end: float, text: str) -> float:
        return x_end - font.text_length(text, 8)

    for y, label, note, cur, prev in rows:
        put((59, y), label, 8)
        if note:
            put((314, y), note, 8)
        put((right(432, cur), y), cur, 8)
        put((right(528, prev), y), prev, 8)
    # A label that wraps onto a second line, with values on the second line.
    put((59, 195), "Özkaynak Yöntemiyle Değerlenen Yatırımların", 8)
    put((59, 206), "Karlarından Paylar", 8)
    put((right(432, "5.919"), 206), "5.919", 8)
    put((right(528, "6.716"), 206), "6.716", 8)
    doc.save(path)


@pytest.fixture
def statement_doc(tmp_path, monkeypatch) -> Document:
    from kap_analyst import config

    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    _write_statement_pdf(tmp_path / "TEST_FS.pdf")
    return Document("TEST_FS", "TEST", "Test A.Ş.", "Test", "financial_statements", "FY2024", "milyon TL",
                    "https://example.org", "-", 1)
