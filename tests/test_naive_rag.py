from hoprag.naive_rag import NaiveRAG


def test_naive_single_retrieval_and_answer(chunks):
    from conftest import FakeRetriever, FakeClaude
    retr = FakeRetriever({"director": chunks})
    claude = FakeClaude(json_responses=[{"answer": "Tim Burton", "cited_chunk_ids": ["c3"]}])
    rag = NaiveRAG(retr, claude, top_k=3)
    res = rag.answer("who is the director")
    assert res.answer == "Tim Burton"
    assert res.cited_chunk_ids == ["c3"]
    assert res.n_retrievals == 1
    assert res.n_claude_calls == 1
    assert len(retr.calls) == 1  # exactly one retrieval, no looping
