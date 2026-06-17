"""Run the Chinese RAG tool over the user's PDFs.

    python scripts/rag_tool.py --docs AI-test            # http://127.0.0.1:5000

Builds a Chinese-embedding LanceDB index over every PDF in --docs, then serves the UI +
/api/ask. Default embedder is bge-small-zh (~100MB); pass --embed-model BAAI/bge-m3 for
stronger multilingual retrieval if the machine has the RAM (bge-m3 is ~2GB and can OOM on
small machines). The LLM is the authenticated `claude` CLI.
"""

import argparse

from hoprag.ingest import load_pdf_chunks
from hoprag.indexer import build_index, make_bge_embed_fn
from hoprag.retriever import Retriever
from hoprag.claude_client import ClaudeClient
from hoprag.rag_app import create_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="AI-test", help="folder of PDFs to index")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--db", default=".lancedb/rag")
    ap.add_argument("--model", default="haiku", help="claude model for answering/scoring")
    ap.add_argument("--embed-model", default="BAAI/bge-small-zh-v1.5",
                    help="embedding model (bge-m3 is stronger but needs ~3GB RAM)")
    args = ap.parse_args()

    print(f"Reading PDFs from {args.docs} ...")
    chunks = load_pdf_chunks(args.docs)
    print(f"  {len(chunks)} chunks. Embedding with {args.embed_model} (downloads on first run)...")
    embed_fn = make_bge_embed_fn(model_name=args.embed_model)
    table = build_index(chunks, embed_fn=embed_fn, db_path=args.db)
    retriever = Retriever(table, embed_fn=embed_fn)
    id2chunk = {c.id: c for c in chunks}
    sources = sorted({c.source_qid for c in chunks})

    app = create_app(retriever, lambda m: ClaudeClient(model=m, timeout=120), id2chunk, sources)
    print(f"Ready: http://{args.host}:{args.port}  ({len(sources)} docs, {len(chunks)} chunks)")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
