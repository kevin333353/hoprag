"""Flask app for the Chinese RAG tool: live question -> naive vs agentic, each answered
in Chinese (prompts_zh) and scored with the RAG Triad (reference-free LLM-judge). Returns
per-side timing, retrieved sources (file + page), and the agentic hop trace.

naive and agentic run in parallel threads, and the two judge calls run in parallel, to
reduce wall-clock (each `claude -p` call has heavy per-process startup).
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from hoprag.agentic_rag import AgenticRAG, AgenticConfig
from hoprag.naive_rag import NaiveRAG
from hoprag import prompts_zh
from hoprag.rag_eval import score_triad

_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "demo"
TOP_K = 4
# cap the judge prompt size: at most MAX_JUDGE_CONTEXTS retrieved chunks, each truncated to _CTX_CHARS chars
MAX_JUDGE_CONTEXTS = 6
_CTX_CHARS = 320


def _src(cid: str, id2chunk: dict) -> dict:
    c = id2chunk.get(cid)
    return {"id": cid, "title": c.title if c else cid, "text": c.text if c else ""}


def _payload(res, id2chunk: dict, elapsed: float) -> dict:
    return {
        "answer": res.answer,
        "n_claude_calls": res.n_claude_calls,
        "n_retrievals": res.n_retrievals,
        "elapsed_sec": round(elapsed, 1),
        "cited": [_src(cid, id2chunk) for cid in res.cited_chunk_ids],
        "trace": [
            {"query": s.query,
             "retrieved": [_src(cid, id2chunk) for cid in s.retrieved_ids],
             "reasoning": s.reasoning, "sufficient": s.sufficient}
            for s in res.trace
        ],
    }


def _contexts(res, id2chunk: dict) -> list[str]:
    ids, seen = [], set()
    for s in res.trace:
        for cid in s.retrieved_ids:
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
    out = [id2chunk[c].text[:_CTX_CHARS] for c in ids if c in id2chunk]
    return out[:MAX_JUDGE_CONTEXTS]


def _timed(pipe, question):
    t0 = time.monotonic()
    res = pipe.answer(question)
    return res, time.monotonic() - t0


def answer_and_score(retriever, qa_claude, judge_claude, question, id2chunk) -> dict:
    naive = NaiveRAG(retriever, qa_claude, top_k=TOP_K, prompts_mod=prompts_zh)
    agentic = AgenticRAG(retriever, qa_claude, AgenticConfig(top_k=TOP_K), prompts_mod=prompts_zh)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fn, fa = ex.submit(_timed, naive, question), ex.submit(_timed, agentic, question)
        naive_res, naive_t = fn.result()
        agentic_res, agentic_t = fa.result()

    with ThreadPoolExecutor(max_workers=2) as ex:
        sn = ex.submit(score_triad, judge_claude, question, naive_res.answer, _contexts(naive_res, id2chunk))
        sa = ex.submit(score_triad, judge_claude, question, agentic_res.answer, _contexts(agentic_res, id2chunk))
        naive_scores, agentic_scores = sn.result(), sa.result()

    naive_payload, agentic_payload = _payload(naive_res, id2chunk, naive_t), _payload(agentic_res, id2chunk, agentic_t)
    naive_payload["scores"], agentic_payload["scores"] = naive_scores, agentic_scores
    return {"naive": naive_payload, "agentic": agentic_payload}


def create_app(retriever, claude_factory, id2chunk, sources=None):
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(_FRONTEND_DIR, "rag.html")

    @app.get("/api/info")
    def info():
        return jsonify({"sources": sources or [], "n_chunks": len(id2chunk)})

    @app.post("/api/ask")
    def ask():
        data = request.get_json(force=True, silent=True) or {}
        q = (data.get("question") or "").strip()
        model = data.get("model") or "haiku"
        if not q:
            return jsonify({"error": "請輸入問題"}), 400
        claude = claude_factory(model)
        t0 = time.monotonic()
        try:
            result = answer_and_score(retriever, claude, claude, q, id2chunk)
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 502
        result["question"] = q
        result["model"] = model
        result["total_sec"] = round(time.monotonic() - t0, 1)
        return jsonify(result)

    return app
