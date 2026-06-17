from hoprag.agentic_rag import AgenticRAG, AgenticConfig
from hoprag.types import Chunk

C1 = [Chunk(id="c1", title="Ed Wood", text="Ed Wood is a 1994 film directed by Tim Burton.")]
C2 = [Chunk(id="c2", title="Tim Burton", text="Tim Burton was born in Burbank.")]


def make(json_responses, retr_script):
    from conftest import FakeRetriever, FakeClaude
    return AgenticRAG(FakeRetriever(retr_script), FakeClaude(json_responses=json_responses))


def test_full_loop_two_hops_then_synth_then_verify():
    # decompose -> hop1(insufficient, next="Tim Burton birthplace") -> hop2(sufficient)
    # -> synth -> verify(supported)
    responses = [
        {"first_query": "director of Ed Wood", "subquestions": []},
        {"reasoning": "found director", "sufficient": False, "next_query": "Tim Burton birthplace"},
        {"reasoning": "found birthplace", "sufficient": True, "next_query": ""},
        {"answer": "Burbank", "cited_chunk_ids": ["c2"]},
        {"supported": True, "revised_answer": "Burbank", "cited_chunk_ids": ["c2"]},
    ]
    rag = make(responses, {"Ed Wood": C1, "birthplace": C2})
    res = rag.answer("Where was the director of Ed Wood born?")
    assert res.answer == "Burbank"
    assert res.n_retrievals == 2
    assert res.n_claude_calls == 5  # decompose + 2 hops + synth + verify
    assert len(res.trace) == 2
    assert res.trace[1].sufficient is True


def test_verify_revises_unsupported_answer():
    responses = [
        {"first_query": "q", "subquestions": []},
        {"reasoning": "enough", "sufficient": True, "next_query": ""},
        {"answer": "Hollywood", "cited_chunk_ids": ["c2"]},
        {"supported": False, "revised_answer": "Burbank", "cited_chunk_ids": ["c2"]},
    ]
    rag = make(responses, {"q": C2})
    res = rag.answer("q")
    assert res.answer == "Burbank"  # revised


def test_ablation_no_decompose_skips_first_call_and_uses_question():
    responses = [
        {"reasoning": "enough", "sufficient": True, "next_query": ""},
        {"answer": "Burbank", "cited_chunk_ids": ["c2"]},
        {"supported": True, "revised_answer": "Burbank", "cited_chunk_ids": ["c2"]},
    ]
    from conftest import FakeRetriever, FakeClaude
    retr = FakeRetriever({"raw question": C2})
    rag = AgenticRAG(retr, FakeClaude(json_responses=responses),
                     AgenticConfig(decompose=False, verify_citations=True))
    res = rag.answer("raw question")
    assert retr.calls[0] == "raw question"  # used question verbatim, no decompose
    assert res.n_claude_calls == 3          # no decompose call


def test_ablation_fixed_hops_ignores_sufficiency():
    # sufficiency_check off, fixed_hops=2 -> must do exactly 2 hops even if model says sufficient
    responses = [
        {"first_query": "q", "subquestions": []},
        {"reasoning": "x", "sufficient": True, "next_query": "second"},
        {"reasoning": "y", "sufficient": True, "next_query": ""},
        {"answer": "A", "cited_chunk_ids": []},
    ]
    from conftest import FakeRetriever, FakeClaude
    retr = FakeRetriever({"q": C1, "second": C2})
    rag = AgenticRAG(retr, FakeClaude(json_responses=responses),
                     AgenticConfig(sufficiency_check=False, fixed_hops=2,
                                   verify_citations=False))
    res = rag.answer("anything")
    assert res.n_retrievals == 2


def test_max_hops_caps_loop():
    # always insufficient -> must stop at max_hops
    hop = {"reasoning": "more", "sufficient": False, "next_query": "again"}
    responses = [{"first_query": "again", "subquestions": []}] + [hop] * 3 + \
                [{"answer": "A", "cited_chunk_ids": []}]
    from conftest import FakeRetriever, FakeClaude
    retr = FakeRetriever({"again": C1})
    rag = AgenticRAG(retr, FakeClaude(json_responses=responses),
                     AgenticConfig(max_hops=3, verify_citations=False))
    res = rag.answer("q")
    assert res.n_retrievals == 3


def test_config_rejects_fixed_hops_above_max_hops():
    import pytest
    with pytest.raises(ValueError):
        AgenticConfig(fixed_hops=10, max_hops=5)


def test_loop_tolerates_null_next_query_and_missing_citations():
    # real models return next_query: null when sufficient, and may omit cited ids
    responses = [
        {"first_query": "q", "subquestions": []},
        {"reasoning": "enough", "sufficient": True, "next_query": None},
        {"answer": "Burbank"},   # no cited_chunk_ids key
        {"supported": True},     # no revised_answer / cited_chunk_ids keys
    ]
    from conftest import FakeRetriever, FakeClaude
    retr = FakeRetriever({"q": C2})
    rag = AgenticRAG(retr, FakeClaude(json_responses=responses))
    res = rag.answer("q")
    assert res.answer == "Burbank"
    assert res.cited_chunk_ids == []
