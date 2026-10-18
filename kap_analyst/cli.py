"""Command-line entry point: ``kap-analyst <command>``."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

from . import config


def cmd_download(args: argparse.Namespace) -> None:
    from .ingest import download, load_manifest

    download(load_manifest(), force=args.force)


def cmd_verify(args: argparse.Namespace) -> None:
    from .ingest import load_manifest
    from .labels import load_qa, verify_labels

    items = load_qa()
    problems = verify_labels(items, load_manifest())
    for p in problems:
        print("✗", p)
    print(f"{len(items) - len({p.split(':')[0] for p in problems})}/{len(items)} QA labels verified against the PDFs")
    sys.exit(1 if problems else 0)


def cmd_index(args: argparse.Namespace) -> None:
    from .ingest import build_chunks, load_manifest, save_chunks
    from .retrieval import Dense, embedding_cache

    docs = load_manifest()
    t0 = time.perf_counter()
    chunks = build_chunks(docs)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    save_chunks(chunks, config.INDEX_DIR / "chunks.jsonl")
    for old in config.INDEX_DIR.glob("embeddings-*.npy"):
        old.unlink()  # chunks changed, so every cached embedding matrix is stale
    Dense(chunks, config.EMBED_MODEL, embedding_cache(config.EMBED_MODEL))
    meta = {"embed_model": config.EMBED_MODEL, "chunks": len(chunks), "docs": [d.doc_id for d in docs],
            "chunk_words": config.CHUNK_WORDS, "chunk_overlap": config.CHUNK_OVERLAP}
    (config.INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents in {time.perf_counter() - t0:.1f}s")


def cmd_eval(args: argparse.Namespace) -> None:
    from .agent import guess_ticker
    from .evaluation import evaluate_retriever, relevance_sets, to_json, to_markdown
    from .ingest import load_chunks, load_manifest
    from .labels import load_qa
    from .retrieval import BM25, Dense, Hybrid, Reranked, embedding_cache

    docs = load_manifest()
    items = load_qa()
    relevant = relevance_sets(items, docs)
    chunks = load_chunks(config.INDEX_DIR / "chunks.jsonl")
    meta = json.loads((config.INDEX_DIR / "meta.json").read_text())

    def short(m: str) -> str:
        return m.split("/")[-1]

    bm25_raw, bm25 = BM25(chunks, prefix=None), BM25(chunks, prefix=5)
    baseline = Dense(chunks, config.BASELINE_EMBED_MODEL, embedding_cache(config.BASELINE_EMBED_MODEL))
    dense = Dense(chunks, meta["embed_model"], embedding_cache(meta["embed_model"]))
    hybrid = Hybrid(bm25, dense)
    reranked = Reranked(hybrid)
    variants = [
        ("BM25 (no stemming)", bm25_raw, None),
        ("BM25 + Turkish F5 stemming", bm25, None),
        (f"Dense · {short(config.BASELINE_EMBED_MODEL)}", baseline, None),
        (f"Dense · {short(meta['embed_model'])}", dense, None),
        (f"Hybrid RRF (BM25-F5 + {short(meta['embed_model'])})", hybrid, None),
        (f"Hybrid + rerank · {short(config.RERANK_MODEL)}", reranked, None),
        ("**Hybrid + rerank + company routing (default)**", reranked, guess_ticker),
    ]
    if not args.fast:
        variants.append((f"Hybrid + rerank · {short(config.QUALITY_RERANK_MODEL)} + routing",
                         Reranked(hybrid, config.QUALITY_RERANK_MODEL), guess_ticker))
    reports = []
    for name, retriever, ticker_fn in variants:
        print(f"→ {name}", flush=True)
        reports.append(evaluate_retriever(name, retriever, items, relevant, ticker_fn))

    extraction = eval_extraction(docs)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "retrieval.json").write_text(to_json(reports))
    (config.RESULTS_DIR / "extraction.json").write_text(json.dumps(extraction, ensure_ascii=False, indent=2))
    env = (f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, "
           f"{len(chunks)} chunks, {len(items)} questions")
    md = ["# Evaluation results", "", f"_Environment: {env}_", "", "## Retrieval (page-level)", "",
          to_markdown(reports), "", "### By query language (Hit@5)", "",
          "| Retriever | " + " | ".join(reports[0].by_lang) + " |",
          "|---|" + "---|" * len(reports[0].by_lang)]
    for r in reports:
        md.append(f"| {r.name} | " + " | ".join(f"{v['hit@5']:.3f} (n={v['n']})" for v in r.by_lang.values()) + " |")
    md += ["", "## Statement extraction", "",
           f"Exact-match accuracy: **{extraction['correct']}/{extraction['total']}** values "
           f"({extraction['accuracy']:.1%}); correct source page: "
           f"{extraction['page_correct']}/{extraction['items']} items.",
           ""]
    if extraction["errors"]:
        md += ["| Item | Period | Expected | Got |", "|---|---|---|---|"]
        md += [f"| {e['item']} | {e['period']} | {e['expected']} | {e['got']} |" for e in extraction["errors"]]
    md += ["", "## Ratios (computed from extracted values)", "", "| Ratio | 2024 | Formula |", "|---|---|---|"]
    md += [f"| {r['ratio']} | {r['value']:.4f} | {r['formula'].replace('|', '&#124;')} |" for r in extraction["ratios"]]
    (config.RESULTS_DIR / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def eval_extraction(docs) -> dict:
    from .ratios import RATIOS, compute_ratio
    from .statements import extract_statements, get_line_item

    gold = [json.loads(line) for line in (config.EVAL_DIR / "statement_items.jsonl").read_text().splitlines() if line]
    sets = {d.doc_id: extract_statements(d) for d in docs if d.kind == "financial_statements"}
    total = correct = page_ok = 0
    errors = []
    for g in gold:
        page_hit = True
        for period, expected in g["values"].items():
            total += 1
            try:
                got = get_line_item(sets[g["doc_id"]], g["item"], period)
                ok = abs(got["value"] - expected) < 1e-6
                page_hit &= got["page"] == g["page"]
                value = got["value"]
            except KeyError:
                ok, value, page_hit = False, None, False
            correct += ok
            if not ok:
                errors.append({"item": g["item"], "period": period, "expected": expected, "got": value})
        page_ok += page_hit
    thy = sets.get("THYAO_2024_FS")
    ratios = [compute_ratio(thy, r, "2024") for r in RATIOS] if thy else []
    return {"total": total, "correct": correct, "accuracy": correct / total, "items": len(gold),
            "page_correct": page_ok, "errors": errors, "ratios": ratios}


def cmd_ask(args: argparse.Namespace) -> None:
    import os

    from .agent import Analyst, extractive_answer
    from .ingest import load_manifest
    from .retrieval import load_index
    from .tools import Toolbox

    toolbox = Toolbox(load_index(rerank=not args.no_rerank), load_manifest())
    if os.environ.get("ANTHROPIC_API_KEY") and not args.extractive:
        res = Analyst(toolbox).ask(args.question)
    else:
        res = extractive_answer(toolbox, args.question)
    print(res.answer)
    if res.tool_calls:
        calls = (f"{c['name']}({json.dumps(c['input'], ensure_ascii=False)})" for c in res.tool_calls)
        print("\nTools:", ", ".join(calls))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("kap_analyst.api:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kap-analyst", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download", help="Fetch source PDFs from KAP and verify SHA-256")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_download)
    p = sub.add_parser("verify-labels", help="Check every QA answer is printed on its gold page")
    p.set_defaults(fn=cmd_verify)
    sub.add_parser("index", help="Chunk documents and build BM25 + embedding index").set_defaults(fn=cmd_index)
    p = sub.add_parser("eval", help="Run retrieval ablation + statement-extraction benchmark")
    p.add_argument("--fast", action="store_true", help="Skip the slow large-reranker variant")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("ask", help="Ask a question (agent with ANTHROPIC_API_KEY, extractive otherwise)")
    p.add_argument("question")
    p.add_argument("--extractive", action="store_true")
    p.add_argument("--no-rerank", action="store_true")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("serve", help="Start the web UI + REST API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
