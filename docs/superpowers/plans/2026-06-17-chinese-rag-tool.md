# Chinese RAG Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Traditional-Chinese RAG tool: ask live questions over the user's own PDFs and see naive vs agentic answers side-by-side, each with RAG-Triad quality scores, timing, and source citations (file + page).

**Architecture:** Reuse the v1 agentic loop with injected Chinese prompts (`prompts_zh`). Ingest PDFs (PyMuPDF) into a bge-m3 LanceDB index. A Flask app runs naive ‖ agentic in parallel threads, scores both with a reference-free RAG-Triad LLM judge, and returns timing + sources + trace. A Traditional-Chinese single-page UI renders the comparison.

**Tech Stack:** Python 3.10+, PyMuPDF, sentence-transformers (BAAI/bge-m3), LanceDB, Flask, Claude CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-06-17-chinese-rag-tool-design.md`

---

## Status of spiked code (must be reviewed, not assumed good)

Committed in `1af4b38` (46 tests pass): `src/hoprag/ingest.py`, `src/hoprag/prompts_zh.py`, `src/hoprag/rag_eval.py`, and the `prompts_mod` injection in `naive_rag.py`/`agentic_rag.py` (+ `tests/test_ingest.py`, `tests/test_rag_eval.py`, `tests/test_prompts_zh.py`). `src/hoprag/rag_app.py` is written but **uncommitted and untested**. Task 1 reviews the committed spike; Task 2 tests + commits `rag_app.py`.

## File structure

```
src/hoprag/
  ingest.py          # [spike, committed] PDF -> chunks
  prompts_zh.py      # [spike, committed] Chinese prompts
  rag_eval.py        # [spike, committed] RAG Triad judge
  rag_app.py         # [spike, UNCOMMITTED] Flask app: /api/ask, /api/info
scripts/
  rag_tool.py        # [new] load AI-test PDFs -> bge-m3 index -> serve
demo/
  rag.html           # [new] Traditional-Chinese single-page UI
tests/
  test_rag_app.py    # [new] Flask test_client + fakes
```

---

## Task 1: Code-review the committed spike

No new code. Verify the spike (`1af4b38`) against the spec before building on it.

- [ ] **Step 1: Dispatch a code-quality reviewer** on the diff of `1af4b38` for `ingest.py`, `prompts_zh.py`, `rag_eval.py`, and the injection changes. Check: single responsibility, correctness of `window_text` overlap math, RAG-Triad prompt/score parsing, that injection defaults to English (doesn't break v1 eval), naming, YAGNI.
- [ ] **Step 2: Address any Critical/Important findings**, re-run `pytest -q` (expect 46 passed), commit fixes if any.

---

## Task 2: `rag_app.py` — Flask app (test the spike)

**Files:**
- Implement: `src/hoprag/rag_app.py` (already written as spike — confirm it matches the test below)
- Test: `tests/test_rag_app.py`

The app must: `POST /api/ask {question, model?}` → run `NaiveRAG` and `AgenticRAG` (both with `prompts_mod=prompts_zh`, `top_k=4`) in parallel threads over the injected retriever, score each answer via `rag_eval.score_triad`, and return `{naive, agentic}` where each side has `answer`, `scores` (context_relevance/groundedness/answer_relevance/overall), `elapsed_sec`, `cited` (list of `{id,title,text}`), and `trace`. `GET /api/info` → `{sources, n_chunks}`. Empty question → 400. Pipeline exception → 502 with message. `GET /` serves `demo/rag.html`.

- [ ] **Step 1: Write the failing test** — `tests/test_rag_app.py`

```python
from hoprag.rag_app import create_app
from hoprag import prompts_zh
from hoprag.rag_eval import TRIAD_SCHEMA
from hoprag.types import Chunk


def _id2chunk():
    return {
        "c1": Chunk(id="c1", title="doc p.1", text="人工智慧是讓機器模擬人類智慧的技術。"),
        "c2": Chunk(id="c2", title="doc p.2", text="機器學習是人工智慧的一個分支。"),
    }


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def search(self, query, k):
        return self.chunks[:k]


class SchemaClaude:
    """Routes by schema identity so it is deterministic under parallel threads."""
    def complete_json(self, prompt, schema):
        if schema is TRIAD_SCHEMA:
            return {"context_relevance": 80, "groundedness": 70,
                    "answer_relevance": 90, "rationale": "r"}
        if schema is prompts_zh.DECOMPOSE_SCHEMA:
            return {"first_query": "q", "subquestions": []}
        if schema is prompts_zh.HOP_SCHEMA:
            return {"reasoning": "證據足夠", "sufficient": True, "next_query": None}
        if schema is prompts_zh.SYNTH_SCHEMA:
            return {"answer": "人工智慧是讓機器模擬人類智慧的技術。", "cited_chunk_ids": ["c1"]}
        if schema is prompts_zh.VERIFY_SCHEMA:
            return {"supported": True, "revised_answer": "人工智慧是讓機器模擬人類智慧的技術。",
                    "cited_chunk_ids": ["c1"]}
        raise AssertionError(f"unexpected schema: {schema}")

    def complete(self, prompt):
        return ""


def _app():
    id2chunk = _id2chunk()
    retr = FakeRetriever(id2chunk.values())
    return create_app(retr, lambda m: SchemaClaude(), id2chunk, sources=["doc"])


def test_ask_returns_both_sides_with_scores_timing_sources():
    client = _app().test_client()
    r = client.post("/api/ask", json={"question": "什麼是人工智慧?"})
    assert r.status_code == 200
    d = r.get_json()
    for side in ("naive", "agentic"):
        assert d[side]["answer"].startswith("人工智慧")
        assert d[side]["scores"]["overall"] == 80.0   # (80+70+90)/3
        assert "elapsed_sec" in d[side]
        assert d[side]["cited"][0]["title"] == "doc p.1"
    assert len(d["agentic"]["trace"]) >= 1            # multi-hop trace present
    assert d["naive"]["n_claude_calls"] == 1
    assert "total_sec" in d


def test_ask_requires_question():
    r = _app().test_client().post("/api/ask", json={"question": "   "})
    assert r.status_code == 400


def test_info_endpoint():
    r = _app().test_client().get("/api/info")
    j = r.get_json()
    assert j["sources"] == ["doc"] and j["n_chunks"] == 2
```

- [ ] **Step 2: Run test to verify it fails** (if the spike is absent/mismatched)

Run: `.venv/Scripts/python -m pytest tests/test_rag_app.py -q`
Expected: PASS if the committed spike matches; if it FAILS, fix `rag_app.py` to satisfy the test (do not weaken the test).

- [ ] **Step 3: Confirm the spike implementation** matches `src/hoprag/rag_app.py` as written (parallel `ThreadPoolExecutor` for naive‖agentic and for the two judge calls; `_payload` includes `cited`/`trace`/`elapsed_sec`; `answer_and_score` returns `{naive, agentic}`). If the spike differs from the contract above, reconcile.

- [ ] **Step 4: Run full suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: PASS (49 total — 46 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/hoprag/rag_app.py tests/test_rag_app.py
git commit -m "feat: Chinese RAG Flask app (/api/ask naive vs agentic + RAG-Triad scores, parallel)"
```

---

## Task 3: `scripts/rag_tool.py` — runner

**Files:**
- Create: `scripts/rag_tool.py`

- [ ] **Step 1: Write the script**

```python
"""Run the Chinese RAG tool over the user's PDFs.

    python scripts/rag_tool.py --docs AI-test            # http://127.0.0.1:5000

Builds a bge-m3 LanceDB index over every PDF in --docs (downloads bge-m3 on first run,
~2GB), then serves the UI + /api/ask. The LLM is the authenticated `claude` CLI.
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
    ap.add_argument("--model", default="haiku")
    args = ap.parse_args()

    print(f"Reading PDFs from {args.docs} ...")
    chunks = load_pdf_chunks(args.docs)
    print(f"  {len(chunks)} chunks. Embedding with bge-m3 (downloads ~2GB on first run)...")
    embed_fn = make_bge_embed_fn(model_name="BAAI/bge-m3")
    table = build_index(chunks, embed_fn=embed_fn, db_path=args.db)
    retriever = Retriever(table, embed_fn=embed_fn)
    id2chunk = {c.id: c for c in chunks}
    sources = sorted({c.source_qid for c in chunks})

    app = create_app(retriever, lambda m: ClaudeClient(model=m, timeout=120), id2chunk, sources)
    print(f"Ready: http://{args.host}:{args.port}  ({len(sources)} docs, {len(chunks)} chunks)")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports and wires (no serve, no network)**

Run: `.venv/Scripts/python -c "import ast; ast.parse(open('scripts/rag_tool.py',encoding='utf-8').read()); print('parse OK')"`
Expected: `parse OK`. (Full run is the manual validation in Task 5.)

- [ ] **Step 3: Commit**

```bash
git add scripts/rag_tool.py
git commit -m "feat: rag_tool runner — index user PDFs with bge-m3 and serve"
```

---

## Task 4: `demo/rag.html` — Traditional-Chinese UI

**Files:**
- Create: `demo/rag.html`

Build with the **frontend-design** skill. It is a single self-contained page (HTML+CSS+JS, Google Fonts ok) that talks to the Task 2 API. All visible text in **Traditional Chinese**.

**Requirements (the contract it must satisfy):**
- On load: `GET /api/info` → show the indexed source files + chunk count (e.g. 「已載入 7 份文件 · 1234 段」).
- A question box (中文 placeholder) + model selector (haiku/sonnet/opus) + 「送出」button. Enter submits.
- On submit: `POST /api/ask {question, model}`; show a clear **「處理中…」progress state** with rotating step labels (it takes ~1–2 min) and an elapsed timer.
- On response: two columns side-by-side — **「一般 RAG(單次檢索)」** vs **「Agentic RAG(多跳推理)」**. Each column shows:
  - the Chinese **answer**;
  - three **score bars** (檢索相關性 / 忠實度 / 回答相關性, 0–100) + overall;
  - **作答時間**(`elapsed_sec`)、Claude 呼叫次數;
  - **來源** chips from `cited` (`title` = 檔名 p.頁), hover shows `text`;
  - for agentic, the **多跳 trace**(每跳 query / 檢索到的來源 / reasoning / 是否足夠).
- Plain-language one-liner at top explaining what the user is looking at and what to compare (the score difference).
- Error responses (`{error}`) shown as a friendly message, not a raw stack.

- [ ] **Step 1: Invoke frontend-design** and generate `demo/rag.html` meeting the contract above.
- [ ] **Step 2: Verify it serves** (Flask test_client, no Claude needed):

```python
# quick check
from hoprag.rag_app import create_app
app = create_app(None, lambda m: None, {}, sources=[])
c = app.test_client()
assert c.get("/").status_code == 200
assert "Agentic" in c.get("/").get_data(as_text=True)
```

- [ ] **Step 3: Commit**

```bash
git add demo/rag.html
git commit -m "feat: Traditional-Chinese RAG tool UI (naive vs agentic, scores, sources, trace)"
```

---

## Task 5: Live validation on the user's PDFs (manual)

**Files:** none (runs the real tool).

- [ ] **Step 1: Run the tool** on `AI-test/` and confirm it indexes (downloads bge-m3 first time):

Run: `.venv/Scripts/python scripts/rag_tool.py --docs AI-test`
Expected: prints `Ready: ...` with the doc/chunk counts; no crash.

- [ ] **Step 2: Ask 2–3 real questions** (about the iPAS materials) via the UI or a `test_client` POST with the real retriever+ClaudeClient. Confirm: Chinese answers, sensible RAG-Triad scores, agentic ≥ naive on overall for at least one multi-hop question, correct sources (file+page), timing shown. Record the numbers.
- [ ] **Step 3: Confirm reachability** — open `http://127.0.0.1:5000` in a browser and verify it loads (if localhost is intercepted on this machine, try `--host 0.0.0.0` + the LAN IP, or a different port, and note the working access method).

---

## Task 6: Docs + finish

- [ ] **Step 1: Add a 中文 RAG 工具 section to `README.md`** documenting `python scripts/rag_tool.py --docs <folder>`, the RAG-Triad scores, the latency/cost note, and that user PDFs are gitignored.
- [ ] **Step 2: Full-suite gate** — `.venv/Scripts/python -m pytest -q` → all pass (49).
- [ ] **Step 3: Commit, then use superpowers:finishing-a-development-branch.**

---

## Self-Review (completed by plan author)

**1. Spec coverage**
- §3 `ingest`/`prompts_zh`/`rag_eval`/injection → spike + Task 1 (review). ✓
- §3 `rag_app` → Task 2. ✓  `scripts/rag_tool` → Task 3. ✓  `demo/rag.html` → Task 4. ✓
- §1/§5 live naive‖agentic + scores + timing + sources + trace → Task 2 (API) + Task 4 (UI). ✓
- §2 bge-m3 → Task 3 (`make_bge_embed_fn("BAAI/bge-m3")`). ✓
- §4 RAG-Triad → rag_eval (spike) + Task 2 wires it. ✓
- §6 latency mitigation (parallel + progress UI) → Task 2 (threads) + Task 4 (progress state). ✓
- §7 error handling (502, 400) → Task 2 tests. ✓
- §8 testing → Tasks 1–2 + Task 5 live. ✓
- §9 AI-test gitignored → done (`48e1ece`). ✓
- §10 keep eval/v1 demo, no upload UI → respected (read-from-folder). ✓

**2. Placeholder scan:** No TBD/"handle edge cases"/"similar to". Frontend HTML is generated by frontend-design against an explicit contract (not pasted) — acceptable for a UI task. ✓

**3. Type consistency:** `create_app(retriever, claude_factory, id2chunk, sources)`, `answer_and_score(...)→{naive,agentic}`, payload keys `answer/scores/elapsed_sec/cited/trace/n_claude_calls`, `score_triad→{context_relevance,groundedness,answer_relevance,overall}`, `prompts_zh.*_SCHEMA`, `load_pdf_chunks(folder)`, `make_bge_embed_fn(model_name=...)` — consistent across tasks and with the committed spike. ✓
