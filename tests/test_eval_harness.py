from hoprag.eval_harness import evaluate
from hoprag.dataset import Example
from hoprag.types import Result, Chunk


class StubPipeline:
    """Maps cited ids to chunk titles via a lookup; returns scripted results."""
    def __init__(self, results):
        self.results = list(results)

    def answer(self, question):
        return self.results.pop(0)


def test_evaluate_aggregates_means():
    examples = [
        Example(qid="q1", question="?", answer="Tim Burton", gold_support_titles=["Tim Burton"]),
        Example(qid="q2", question="?", answer="Burbank", gold_support_titles=["Burbank"]),
    ]
    id2title = {"c_tb": "Tim Burton", "c_bb": "Burbank", "c_x": "Other"}
    results = [
        Result(question="?", answer="Tim Burton", cited_chunk_ids=["c_tb"],
               n_claude_calls=5, n_retrievals=2),
        Result(question="?", answer="Hollywood", cited_chunk_ids=["c_x"],
               n_claude_calls=4, n_retrievals=2),
    ]
    report = evaluate(StubPipeline(results), examples, id2title)
    assert report["em"] == 0.5          # q1 exact, q2 wrong
    assert report["support_recall"] == 0.5  # q1 hit, q2 missed
    assert report["avg_claude_calls"] == 4.5
    assert report["n"] == 2
