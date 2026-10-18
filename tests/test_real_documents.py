"""Integration checks against the real KAP PDFs. Skipped unless `kap-analyst download` has been run."""

import json

import pytest

from kap_analyst import config
from kap_analyst.ingest import load_manifest, sha256_file
from kap_analyst.labels import load_qa, verify_labels

DOCS = load_manifest()
pytestmark = pytest.mark.skipif(
    not all(d.path.exists() for d in DOCS), reason="source PDFs not downloaded (run `kap-analyst download`)"
)


def test_checksums_match_manifest():
    for d in DOCS:
        assert sha256_file(d.path) == d.sha256, d.doc_id


def test_every_qa_label_is_printed_on_its_gold_page():
    assert verify_labels(load_qa(), DOCS) == []


def test_statement_extraction_matches_hand_labels():
    from kap_analyst.cli import eval_extraction

    result = eval_extraction(DOCS)
    assert result["accuracy"] == 1.0, result["errors"]


def test_thy_ratios_match_hand_calculation():
    from kap_analyst.ratios import compute_ratio
    from kap_analyst.statements import extract_statements

    thy = extract_statements(next(d for d in DOCS if d.ticker == "THYAO"))
    assert compute_ratio(thy, "net_margin")["value"] == pytest.approx(113_357 / 745_430, abs=1e-6)
    assert compute_ratio(thy, "current_ratio")["value"] == pytest.approx(341_910 / 339_533, abs=1e-6)
    roe = compute_ratio(thy, "roe")
    assert roe["value"] == pytest.approx(113_378 / ((679_887 + 457_257) / 2), abs=1e-6)
    assert {i["page"] for i in roe["inputs"]} == {10, 11}


def test_eval_files_are_well_formed():
    rows = [json.loads(x) for x in (config.EVAL_DIR / "statement_items.jsonl").read_text().splitlines() if x]
    assert len(rows) == 45 and all(set(r["values"]) == {"2024", "2023"} for r in rows)
