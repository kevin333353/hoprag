from hoprag.rag_eval import score_triad, TRIAD_SCHEMA, triad_prompt


class FakeJudge:
    def __init__(self, resp):
        self.resp = resp

    def complete_json(self, prompt, schema):
        assert schema is TRIAD_SCHEMA
        self.prompt = prompt
        return dict(self.resp)


def test_score_triad_adds_overall_mean():
    j = FakeJudge({"context_relevance": 90, "groundedness": 60,
                   "answer_relevance": 75, "rationale": "ok"})
    s = score_triad(j, "Q?", "an answer", ["ctx1", "ctx2"])
    assert s["overall"] == 75.0          # (90+60+75)/3
    assert s["groundedness"] == 60
    assert "Q?" in j.prompt and "ctx1" in j.prompt


def test_triad_schema_required():
    for k in ("context_relevance", "groundedness", "answer_relevance"):
        assert k in TRIAD_SCHEMA["required"]


def test_triad_prompt_handles_no_context():
    p = triad_prompt("Q", "A", [])
    assert "沒有檢索到任何內容" in p
