from hoprag.types import Chunk, HopStep, Result


def test_result_defaults_and_costs():
    r = Result(question="q", answer="a", cited_chunk_ids=["c1"])
    assert r.trace == []
    assert r.n_claude_calls == 0
    assert r.n_retrievals == 0
    r.trace.append(HopStep(query="q1", retrieved_ids=["c1"], reasoning="because", sufficient=True))
    assert r.trace[0].sufficient is True


def test_chunk_fields():
    c = Chunk(id="c1", title="T", text="body")
    assert c.source_qid is None
