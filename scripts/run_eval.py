import argparse
import json
import os

from hoprag.agentic_rag import AgenticRAG, AgenticConfig
from hoprag.naive_rag import NaiveRAG
from hoprag.indexer import build_index, make_bge_embed_fn
from hoprag.retriever import Retriever
from hoprag.dataset import load_hotpotqa, pool_chunks, examples_from_records
from hoprag.eval_harness import evaluate
from hoprag.report import markdown_table, cost_curve_png, traces_markdown


def build_variants(retriever, claude):
    return {
        "naive": NaiveRAG(retriever, claude),
        "agentic_full": AgenticRAG(retriever, claude, AgenticConfig()),
        "ablation_no_decompose": AgenticRAG(retriever, claude, AgenticConfig(decompose=False)),
        "ablation_fixed_hops": AgenticRAG(retriever, claude, AgenticConfig(sufficiency_check=False)),
        "ablation_no_verify": AgenticRAG(retriever, claude, AgenticConfig(verify_citations=False)),
    }


def run(n, db_path, out_dir):
    from hoprag.claude_client import ClaudeClient
    os.makedirs(out_dir, exist_ok=True)
    records = load_hotpotqa(n=n)
    chunks = pool_chunks(records)
    examples = examples_from_records(records)
    id2title = {c.id: c.title for c in chunks}

    embed_fn = make_bge_embed_fn()
    table = build_index(chunks, embed_fn=embed_fn, db_path=db_path)
    retriever = Retriever(table, embed_fn=embed_fn)
    claude = ClaudeClient()

    reports = {}
    for name, pipe in build_variants(retriever, claude).items():
        reports[name] = evaluate(pipe, examples, id2title)
        print(name, {k: reports[name][k] for k in ("em", "f1", "support_recall", "avg_claude_calls")})

    md = markdown_table(reports)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    cost_curve_png(reports, os.path.join(out_dir, "cost_curve.png"))
    with open(os.path.join(out_dir, "traces.md"), "w", encoding="utf-8") as f:
        f.write(traces_markdown(reports["agentic_full"]))
    with open(os.path.join(out_dir, "reports.json"), "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    print("\n" + md)


def smoke():
    """Wiring check with self-contained fakes — no network, no CLI, no test imports."""
    from hoprag.types import Chunk

    class _FakeRetriever:
        def __init__(self, chunks):
            self.chunks = chunks
        def search(self, query, k):
            return self.chunks[:k]

    class _FakeClaude:
        def __init__(self, responses):
            self.responses = list(responses)
        def complete_json(self, prompt, schema):
            return self.responses.pop(0)
        def complete(self, prompt):
            return ""

    c = [Chunk(id="c1", title="Tim Burton", text="filmmaker born in Burbank")]
    retr = _FakeRetriever(c)
    claude = _FakeClaude([
        {"first_query": "q", "subquestions": []},
        {"reasoning": "ok", "sufficient": True, "next_query": ""},
        {"answer": "Burbank", "cited_chunk_ids": ["c1"]},
        {"supported": True, "revised_answer": "Burbank", "cited_chunk_ids": ["c1"]},
    ])
    rag = AgenticRAG(retr, claude)
    res = rag.answer("where was Tim Burton born?")
    assert res.answer == "Burbank"
    print("smoke OK:", res.answer, "calls:", res.n_claude_calls)

    naive = NaiveRAG(_FakeRetriever(c), _FakeClaude([
        {"answer": "Burbank", "cited_chunk_ids": ["c1"]},
    ]))
    nres = naive.answer("where was Tim Burton born?")
    assert nres.answer == "Burbank"
    print("smoke OK (naive):", nres.answer, "calls:", nres.n_claude_calls)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--db", default=".lancedb/hotpot")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        run(args.n, args.db, args.out)
