# HopRAG — Agentic Multi-Hop RAG

An agentic retrieval system that beats naive RAG on multi-hop questions by **orchestrating** retrieval instead of doing it in one shot. The orchestration loop is the point: it is a hand-written, inspectable, *ablatable* strategy — not a generic agent framework — and Claude (via the `claude` CLI) is called as the per-step reasoning model. A fixed local dense retriever is shared across every variant, so any measured accuracy lift is attributable to the loop, not a fancier retriever.

## 中文 RAG 工具 — query your own PDFs

A Traditional-Chinese tool that runs **naive vs agentic side-by-side over your own PDFs**, scores each answer with the reference-free **RAG Triad** (context relevance / groundedness / answer relevance — the TruLens/RAGAS standard, LLM-as-judge, no answer key needed), and shows per-side timing + cited sources (file + page).

```bash
python scripts/rag_tool.py --docs AI-test          # → http://127.0.0.1:5000
# stronger multilingual embeddings (but ~2GB, needs RAM):  --embed-model BAAI/bge-m3
```

- **PDF ingest** (`hoprag.ingest`, PyMuPDF) → overlapping chunks tagged with file + page.
- **Embeddings**: default `BAAI/bge-small-zh-v1.5` (light, Chinese); `bge-m3` opt-in. Downloads use the OS trust store (`truststore`) so they work behind corporate TLS proxies.
- **Answers in Traditional Chinese** via injected `prompts_zh`; the same agentic loop, in Chinese.
- **Validated** on a 7-PDF iPAS corpus — e.g. 「什麼是生成式 AI？它和傳統辨識式 AI 有何不同?」 → naive **89.0** vs agentic **96.0** (grounded, cited 科目1/科目2 with page numbers).
- **Cost/latency**: ~7–9 `claude` calls per question (~1–2 min; per-process CLI startup dominates). Default model `haiku`. Your PDFs stay local — `AI-test/` is gitignored.

## The idea

Naive RAG retrieves once on the raw question and answers. That fails on multi-hop questions where the bridge fact is only findable *after* you've answered the first hop (e.g. "Where was the director of *Ed Wood* born?" — you must first learn the director is Tim Burton, then retrieve Burton's birthplace). HopRAG runs a loop instead:

1. **Decompose / plan** — turn the question into an initial search query (and subquestions).
2. **Multi-hop retrieve** — retrieve, then use what was just learned to form the *next* query.
3. **Sufficiency self-check** — Claude decides whether the gathered evidence is enough, controlling when the loop stops (hard-capped by `max_hops`).
4. **Synthesize with citations** — answer briefly, citing the chunks used.
5. **Verify citations** — check each claim is supported by its cited evidence; revise if not.

```
question
  → (1) decompose / plan ............................ Claude
  → loop (cap max_hops):
        retrieve(current_query) .................... fixed dense retriever
        → (2) hop reasoning -> next_query .......... Claude
        → (3) sufficiency self-check (stop?) ....... Claude
  → (4) synthesize answer + citations ............... Claude
  → (5) verify citations (revise if unsupported) .... Claude
  → answer + cited chunks + full trace
```

Each numbered mechanism can be toggled off via `AgenticConfig`, which is how the ablations are built.

## Architecture

| Module | Responsibility |
| --- | --- |
| `hoprag.types` | `Chunk`, `HopStep`, `Result` data model |
| `hoprag.dataset` | Load HotpotQA, pool context paragraphs into a deduped corpus, extract gold answers/supporting titles |
| `hoprag.indexer` / `hoprag.retriever` | Build a LanceDB index from an injectable `embed_fn`; fixed dense search shared by all variants |
| `hoprag.claude_client` | Wrap `claude -p --output-format json`; `complete` / `complete_json` (schema-validated, retried) |
| `hoprag.prompts` | Prompt builders + JSON schemas for the decompose / hop / synth / verify steps |
| `hoprag.naive_rag` | B0 baseline: one retrieval, one answer |
| `hoprag.agentic_rag` | The orchestration loop + `AgenticConfig` toggles (full system and all ablations) |
| `hoprag.eval_harness` | Run any pipeline over examples; aggregate EM/F1 + support recall + cost |
| `hoprag.report` | Markdown comparison table + accuracy-vs-cost plot |
| `scripts/run_eval.py` | Operator entry point (`--smoke` wiring check, or a real run) |

## Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

(On macOS/Linux use `.venv/bin/python`.)

## Interactive demo

A web UI that runs a question through **naive vs agentic side by side** and visualizes the
agentic multi-hop trace — each hop's query, the documents it retrieved, its reasoning, and the
sufficiency check — revealed step by step.

```bash
.venv/Scripts/python scripts/demo_server.py      # http://127.0.0.1:5000
```

It indexes a small bundled corpus (`hoprag.demo_corpus`) so it starts instantly — no HotpotQA
download. The example questions are precomputed (`scripts/precompute_examples.py` →
`demo/examples_cache.json`) and replay instantly; custom questions run live through the `claude`
CLI (~4–8 calls, ~1 min). Default model is `haiku` to keep cost low.

## Run

Wiring smoke test — no network, no Claude CLI:

```bash
.venv/Scripts/python scripts/run_eval.py --smoke
# -> smoke OK: Burbank calls: 4
```

Full eval — needs internet for HotpotQA + the BGE model on first run, **and an authenticated `claude` CLI on PATH**:

```bash
.venv/Scripts/python scripts/run_eval.py --n 20  --out reports/dev    # small dev run first
.venv/Scripts/python scripts/run_eval.py --n 300 --out reports/full   # headline run
```

This evaluates five variants — `naive`, `agentic_full`, and three ablations (`ablation_no_decompose`, `ablation_fixed_hops`, `ablation_no_verify`) — on a pooled HotpotQA corpus, against the same fixed retriever, and writes `results.md`, `cost_curve.png`, and `reports.json` to the output dir.

> The LLM is the `claude` CLI (`claude -p --output-format json`); it must be installed and signed in. It authenticates via your Claude Code login — no API key needed (a stray/invalid `ANTHROPIC_API_KEY` in the environment is dropped automatically, since it would otherwise 401). Embeddings are local (`BAAI/bge-small-en-v1.5`), so only generation needs the CLI. Each agentic question makes ~4–8 Claude calls, so start with a small `--n`.

## Metrics

- **Answer EM / F1** — HotpotQA official answer normalization.
- **Supporting-fact recall** — did the cited chunks include the gold supporting paragraphs.
- **Avg Claude calls / question** — cost, plotted against F1 in `cost_curve.png`.

## Results

_To be populated by a real eval run._ After running `scripts/run_eval.py`, paste the generated `results.md` table here and embed `cost_curve.png`. Expected shape: `agentic_full` exceeds `naive` on EM/F1 and supporting-fact recall, at the cost of more Claude calls per question; each ablation quantifies one mechanism's contribution.

## Tests

```bash
.venv/Scripts/python -m pytest -q
```

## Development

Built spec-first via the superpowers workflow: brainstorm → spec → plan → TDD, each unit implemented test-first with reviews. See `docs/superpowers/specs/` and `docs/superpowers/plans/`.
