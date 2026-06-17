from hoprag.report import markdown_table, cost_curve_png, traces_markdown


REPORTS = {
    "naive": {"em": 0.40, "f1": 0.52, "support_recall": 0.55,
              "support_precision": 0.30, "avg_claude_calls": 1.0, "avg_retrievals": 1.0},
    "agentic_full": {"em": 0.58, "f1": 0.70, "support_recall": 0.81,
                     "support_precision": 0.40, "avg_claude_calls": 6.0, "avg_retrievals": 3.0},
}


def test_markdown_table_has_rows_and_headers():
    md = markdown_table(REPORTS)
    assert "| variant |" in md.lower()
    assert "naive" in md and "agentic_full" in md
    assert "0.58" in md  # agentic em rendered
    assert "support_precision" in md and "avg_retrievals" in md


def test_cost_curve_writes_png(tmp_path):
    out = tmp_path / "curve.png"
    cost_curve_png(REPORTS, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_traces_markdown_renders_hops():
    report = {"per_question": [
        {"qid": "q1", "answer": "Burbank", "trace": [
            {"query": "director of Ed Wood", "retrieved_ids": ["c1"],
             "reasoning": "found director", "sufficient": False},
            {"query": "Tim Burton birthplace", "retrieved_ids": ["c2"],
             "reasoning": "found birthplace", "sufficient": True},
        ]},
    ]}
    md = traces_markdown(report)
    assert "q1" in md
    assert "Tim Burton birthplace" in md
    assert "hop 1" in md
