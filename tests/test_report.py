from hoprag.report import markdown_table, cost_curve_png


REPORTS = {
    "naive": {"em": 0.40, "f1": 0.52, "support_recall": 0.55, "avg_claude_calls": 1.0},
    "agentic_full": {"em": 0.58, "f1": 0.70, "support_recall": 0.81, "avg_claude_calls": 6.0},
}


def test_markdown_table_has_rows_and_headers():
    md = markdown_table(REPORTS)
    assert "| variant |" in md.lower()
    assert "naive" in md and "agentic_full" in md
    assert "0.58" in md  # agentic em rendered


def test_cost_curve_writes_png(tmp_path):
    out = tmp_path / "curve.png"
    cost_curve_png(REPORTS, str(out))
    assert out.exists() and out.stat().st_size > 0
