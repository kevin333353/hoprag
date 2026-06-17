from hoprag.metrics import exact_match, f1_score, support_recall, support_precision
from hoprag.types import Result


def _retrieved_titles(res, id2title: dict) -> list[str]:
    ids = {cid for step in res.trace for cid in step.retrieved_ids}
    return [id2title.get(cid, "") for cid in ids]


def evaluate(pipeline, examples, id2title: dict) -> dict:
    """Run pipeline over examples; return aggregate metrics. id2title maps chunk id -> title.

    Supporting-fact recall/precision are computed from RETRIEVED chunks (the union of
    retrieved_ids across the result's trace), per spec — they measure whether the gold
    paragraphs were retrieved, independent of how many ids the model chose to cite.
    citation_recall is reported separately for reference. A pipeline error on one
    question is recorded and scored as a miss rather than aborting the whole run.
    """
    n = len(examples)
    em = f1 = sr = sp = cr = calls = retr = 0.0
    n_errors = 0
    per_question = []
    for ex in examples:
        try:
            res = pipeline.answer(ex.question)
            err = None
        except Exception as e:  # degrade: one bad question must not kill the run
            res = Result(question=ex.question, answer="", cited_chunk_ids=[])
            err = str(e)
            n_errors += 1
        retrieved_titles = _retrieved_titles(res, id2title)
        cited_titles = [id2title.get(cid, "") for cid in res.cited_chunk_ids]
        q_em = exact_match(res.answer, ex.answer)
        q_f1 = f1_score(res.answer, ex.answer)
        q_sr = support_recall(retrieved_titles, ex.gold_support_titles)
        q_sp = support_precision(retrieved_titles, ex.gold_support_titles)
        q_cr = support_recall(cited_titles, ex.gold_support_titles)
        em += q_em; f1 += q_f1; sr += q_sr; sp += q_sp; cr += q_cr
        calls += res.n_claude_calls; retr += res.n_retrievals
        per_question.append({
            "qid": ex.qid,
            "em": q_em, "f1": q_f1,
            "support_recall": q_sr, "support_precision": q_sp,
            "citation_recall": q_cr,
            "answer": res.answer, "error": err,
            "trace": [
                {"query": s.query, "retrieved_ids": s.retrieved_ids,
                 "reasoning": s.reasoning, "sufficient": s.sufficient}
                for s in res.trace
            ],
        })
    return {
        "n": n,
        "n_errors": n_errors,
        "em": em / n,
        "f1": f1 / n,
        "support_recall": sr / n,
        "support_precision": sp / n,
        "citation_recall": cr / n,
        "avg_claude_calls": calls / n,
        "avg_retrievals": retr / n,
        "per_question": per_question,
    }
