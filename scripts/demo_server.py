"""Run the interactive demo server.

Builds a LanceDB index over the small bundled demo corpus (downloads the BGE model on
first run) and serves the demo UI + /api/ask. The LLM is the authenticated `claude` CLI;
default per-request model is cheap (haiku) since each question makes ~5-9 calls.

    python scripts/demo_server.py            # http://127.0.0.1:5000
    python scripts/demo_server.py --port 8000
"""

import argparse

from hoprag.indexer import build_index, make_bge_embed_fn
from hoprag.retriever import Retriever
from hoprag.claude_client import ClaudeClient
from hoprag.demo_corpus import demo_chunks
from hoprag.server import create_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--db", default=".lancedb/demo")
    args = ap.parse_args()

    chunks = demo_chunks()
    print(f"Indexing {len(chunks)} demo documents (downloads BGE model on first run)...")
    embed_fn = make_bge_embed_fn()
    table = build_index(chunks, embed_fn=embed_fn, db_path=args.db)
    retriever = Retriever(table, embed_fn=embed_fn)
    id2chunk = {c.id: c for c in chunks}

    app = create_app(retriever, lambda m: ClaudeClient(model=m, timeout=90), id2chunk)
    print(f"Demo ready at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
