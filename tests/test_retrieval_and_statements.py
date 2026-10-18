import pytest

from kap_analyst.ingest import chunk_page
from kap_analyst.retrieval import unique_pages
from kap_analyst.statements import extract_statements, get_line_item


def test_bm25_matches_inflected_turkish_query(bm25):
    # "mağazası" / "açtı" never appear verbatim; F5 stemming maps them onto "mağaza" / "açılışı".
    hits = bm25.search("BİM kaç yeni mağazası açtı", k=3)
    assert hits[0].chunk.ticker == "BIMAS" and hits[0].chunk.page == 19


def test_bm25_ticker_filter(bm25):
    hits = bm25.search("milyon", k=5, ticker="ASELS")
    assert hits == [] or all(h.chunk.ticker == "ASELS" for h in hits)


def test_unique_pages_collapses_chunks(bm25):
    hits = bm25.search("gıda atığı", k=5)
    pages = unique_pages(hits)
    assert len(pages) == len(set(pages))


def test_chunk_page_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    pieces = chunk_page(text, words=100, overlap=20)
    assert pieces[0].split()[-20:] == pieces[1].split()[:20]
    assert pieces[-1].split()[-1] == "w299"


def test_statement_extraction_reads_values_and_negatives(statement_doc):
    stmts = extract_statements(statement_doc)
    assert get_line_item(stmts, "revenue", "2024")["value"] == 745430
    assert get_line_item(stmts, "revenue", "2023")["value"] == 504398
    assert get_line_item(stmts, "cost_of_sales", "2024")["value"] == -602522
    assert get_line_item(stmts, "gross_profit", "2023")["value"] == 119446


def test_statement_extraction_stitches_wrapped_labels(statement_doc):
    stmts = extract_statements(statement_doc)
    item = get_line_item(stmts, "Özkaynak Yöntemiyle Değerlenen Yatırımların Karlarından Paylar", "2024")
    assert item["value"] == 5919 and item["page"] == 1


def test_missing_item_raises(statement_doc):
    with pytest.raises(KeyError):
        get_line_item(extract_statements(statement_doc), "total_assets", "2024")
