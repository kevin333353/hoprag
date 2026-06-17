"""Precompute the demo's example questions so the UI can replay them instantly.

Runs each EXAMPLE_QUESTION through the real pipelines (downloads the BGE model on first
run; makes real `claude` CLI calls) and writes demo/examples_cache.json, which the server
serves from /api/examples. Re-run this to refresh the cached traces.

    python scripts/precompute_examples.py            # uses haiku
    python scripts/precompute_examples.py --model sonnet
"""

import argparse
import json
from pathlib import Path

from hoprag.indexer import build_index, make_bge_embed_fn
from hoprag.retriever import Retriever
from hoprag.claude_client import ClaudeClient
from hoprag.demo_corpus import demo_chunks, EXAMPLE_QUESTIONS
from hoprag.server import run_question

_OUT = Path(__file__).resolve().parents[1] / "demo" / "examples_cache.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--db", default=".lancedb/demo")
    args = ap.parse_args()

    chunks = demo_chunks()
    embed_fn = make_bge_embed_fn()
    table = build_index(chunks, embed_fn=embed_fn, db_path=args.db)
    retriever = Retriever(table, embed_fn=embed_fn)
    id2chunk = {c.id: c for c in chunks}
    claude = ClaudeClient(model=args.model, timeout=90)

    examples = []
    for i, ex in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"[{i}/{len(EXAMPLE_QUESTIONS)}] {ex['question']}")
        payload = run_question(retriever, claude, ex["question"], id2chunk)
        payload["answer"] = ex["answer"]  # gold answer for correctness display
        print(f"    naive={payload['naive']['answer']!r}  "
              f"agentic={payload['agentic']['answer']!r}  "
              f"hops={len(payload['agentic']['trace'])}")
        examples.append(payload)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps({"cached": True, "model": args.model, "examples": examples},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {_OUT}")


if __name__ == "__main__":
    main()
