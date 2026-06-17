from hoprag.server import create_app
from hoprag.types import Chunk


def _chunks():
    return [
        Chunk(id="d0", title="Ed Wood (film)", text="Ed Wood is directed by Tim Burton."),
        Chunk(id="d1", title="Tim Burton", text="Tim Burton was born in Burbank."),
    ]


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k):
        return self.chunks[:k]


class FakeClaude:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete_json(self, prompt, schema):
        return self.responses.pop(0)

    def complete(self, prompt):
        return ""


def _app(responses):
    chunks = _chunks()
    id2chunk = {c.id: c for c in chunks}
    # one claude instance per request, shared by naive + agentic -> script in order:
    # naive synth, then agentic decompose, hop, synth, verify
    return create_app(FakeRetriever(chunks), lambda m: FakeClaude(responses), id2chunk)


def test_ask_returns_naive_and_agentic():
    responses = [
        {"answer": "Burbank", "cited_chunk_ids": ["d1"]},
        {"first_query": "director of Ed Wood", "subquestions": []},
        {"reasoning": "found director then birthplace", "sufficient": True, "next_query": ""},
        {"answer": "Burbank", "cited_chunk_ids": ["d1"]},
        {"supported": True, "revised_answer": "Burbank", "cited_chunk_ids": ["d1"]},
    ]
    client = _app(responses).test_client()
    r = client.post("/api/ask", json={"question": "Where was the director of Ed Wood born?"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["naive"]["answer"] == "Burbank"
    assert data["agentic"]["answer"] == "Burbank"
    assert len(data["agentic"]["trace"]) == 1
    assert data["agentic"]["trace"][0]["sufficient"] is True
    assert "d1" in data["chunks"]
    assert data["chunks"]["d1"]["title"] == "Tim Burton"


def test_ask_requires_question():
    client = _app([]).test_client()
    r = client.post("/api/ask", json={"question": "   "})
    assert r.status_code == 400


def test_examples_endpoint():
    client = _app([]).test_client()
    r = client.get("/api/examples")
    assert r.status_code == 200
    assert len(r.get_json()["examples"]) >= 1
