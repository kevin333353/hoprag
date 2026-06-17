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
