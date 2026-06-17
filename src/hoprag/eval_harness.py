from hoprag.metrics import exact_match, f1_score, support_recall


def evaluate(pipeline, examples, id2title: dict) -> dict:
    """Run pipeline over examples; return aggregate metrics. id2title maps chunk id -> title."""
    n = len(examples)
    em = f1 = sr = calls = retr = 0.0
    per_question = []
    for ex in examples:
        res = pipeline.answer(ex.question)
        cited_titles = [id2title.get(cid, "") for cid in res.cited_chunk_ids]
        q_em = exact_match(res.answer, ex.answer)
        q_f1 = f1_score(res.answer, ex.answer)
        q_sr = support_recall(cited_titles, ex.gold_support_titles)
        em += q_em; f1 += q_f1; sr += q_sr
        calls += res.n_claude_calls; retr += res.n_retrievals
        per_question.append({"qid": ex.qid, "em": q_em, "f1": q_f1,
                             "support_recall": q_sr, "answer": res.answer})
    return {
        "n": n,
        "em": em / n,
        "f1": f1 / n,
        "support_recall": sr / n,
        "avg_claude_calls": calls / n,
        "avg_retrievals": retr / n,
        "per_question": per_question,
    }
