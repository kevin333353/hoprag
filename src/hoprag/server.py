"""Flask backend for the interactive demo.

`create_app` takes injected dependencies (a retriever, a `claude_factory(model)->client`,
and an id->Chunk map) so it can be unit-tested with fakes — no real CLI, no model download.
`scripts/demo_server.py` wires the real BGE retriever + ClaudeClient.

Endpoints:
  GET  /                -> the demo page (demo/index.html)
  GET  /api/examples    -> precomputed example results if demo/examples_cache.json exists,
                           else the bare example questions (run live)
  POST /api/ask         -> {question, model?} runs naive + agentic live, returns both + traces
"""

import json
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from hoprag.agentic_rag import AgenticRAG, AgenticConfig
from hoprag.naive_rag import NaiveRAG
from hoprag.demo_corpus import EXAMPLE_QUESTIONS

_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "demo"
_CACHE_FILE = _FRONTEND_DIR / "examples_cache.json"

# Focused retrieval so multi-hop actually matters on the small demo corpus
# (a large top_k would let a single naive retrieval already pull in the bridge doc).
DEMO_TOP_K = 3


def _result_payload(res) -> dict:
    return {
        "answer": res.answer,
        "cited_chunk_ids": res.cited_chunk_ids,
        "n_claude_calls": res.n_claude_calls,
        "n_retrievals": res.n_retrievals,
        "trace": [
            {"query": s.query, "retrieved_ids": s.retrieved_ids,
             "reasoning": s.reasoning, "sufficient": s.sufficient}
            for s in res.trace
        ],
    }


def run_question(retriever, claude, question: str, id2chunk: dict, top_k: int = DEMO_TOP_K) -> dict:
    """Run both pipelines on one question and build the wire payload. Shared by the
    /api/ask endpoint and the offline example precompute script."""
    naive_res = NaiveRAG(retriever, claude, top_k=top_k).answer(question)
    agentic_res = AgenticRAG(retriever, claude, AgenticConfig(top_k=top_k)).answer(question)

    referenced = set(naive_res.cited_chunk_ids) | set(agentic_res.cited_chunk_ids)
    for res in (naive_res, agentic_res):
        for step in res.trace:
            referenced.update(step.retrieved_ids)
    chunks = {cid: {"title": id2chunk[cid].title, "text": id2chunk[cid].text}
              for cid in referenced if cid in id2chunk}

    return {
        "question": question,
        "naive": _result_payload(naive_res),
        "agentic": _result_payload(agentic_res),
        "chunks": chunks,
    }


def create_app(retriever, claude_factory, id2chunk: dict):
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(_FRONTEND_DIR, "index.html")

    @app.get("/api/examples")
    def examples():
        if _CACHE_FILE.exists():
            return jsonify(json.loads(_CACHE_FILE.read_text(encoding="utf-8")))
        return jsonify({"cached": False, "examples": EXAMPLE_QUESTIONS})

    @app.post("/api/ask")
    def ask():
        data = request.get_json(force=True, silent=True) or {}
        question = (data.get("question") or "").strip()
        model = data.get("model") or "haiku"
        if not question:
            return jsonify({"error": "question is required"}), 400

        claude = claude_factory(model)
        t0 = time.monotonic()
        try:
            payload = run_question(retriever, claude, question, id2chunk)
        except Exception as e:  # surface model/CLI errors to the UI instead of a 500 page
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 502
        payload["model"] = model
        payload["elapsed_sec"] = round(time.monotonic() - t0, 1)
        return jsonify(payload)

    return app
