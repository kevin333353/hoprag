from hoprag.eval_harness import evaluate
from hoprag.dataset import Example
from hoprag.types import Result, HopStep


class StubPipeline:
    def __init__(self, results):
        self.results = list(results)

    def answer(self, question):
        return self.results.pop(0)


class BoomPipeline:
    def answer(self, question):
        raise RuntimeError("boom")


def test_evaluate_aggregates_means():
    examples = [
        Example(qid="q1", question="?", answer="Tim Burton", gold_support_titles=["Tim Burton"]),
        Example(qid="q2", question="?", answer="Burbank", gold_support_titles=["Burbank"]),
    ]
    id2title = {"c_tb": "Tim Burton", "c_bb": "Burbank", "c_x": "Other"}
    results = [
        Result(question="?", answer="Tim Burton", cited_chunk_ids=["c_tb"],
               trace=[HopStep(query="?", retrieved_ids=["c_tb"], reasoning="", sufficient=True)],
               n_claude_calls=5, n_retrievals=2),
        Result(question="?", answer="Hollywood", cited_chunk_ids=["c_x"],
               trace=[HopStep(query="?", retrieved_ids=["c_x"], reasoning="", sufficient=True)],
               n_claude_calls=4, n_retrievals=2),
    ]
    report = evaluate(StubPipeline(results), examples, id2title)
    assert report["em"] == 0.5                # q1 exact, q2 wrong
    assert report["support_recall"] == 0.5    # q1 retrieved gold, q2 did not
    assert report["support_precision"] == 0.5  # q1 1/1, q2 0/1
    assert report["citation_recall"] == 0.5
    assert report["avg_claude_calls"] == 4.5
    assert report["avg_retrievals"] == 2.0
    assert report["n"] == 2
    assert report["n_errors"] == 0


def test_evaluate_degrades_on_pipeline_error():
    examples = [Example(qid="q1", question="?", answer="X", gold_support_titles=["T"])]
    report = evaluate(BoomPipeline(), examples, {})
    assert report["n_errors"] == 1
    assert report["em"] == 0.0
    assert report["per_question"][0]["error"] == "boom"
