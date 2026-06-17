from hoprag.metrics import normalize_answer, exact_match, f1_score, support_recall


def test_normalize_strips_articles_punct_case():
    assert normalize_answer("The Dark Knight!") == "dark knight"


def test_exact_match():
    assert exact_match("the answer", "Answer") == 1.0
    assert exact_match("wrong", "right") == 0.0


def test_f1_partial_overlap():
    # pred 2 tokens, gold 3 tokens, 2 common -> p=1.0, r=2/3 -> f1=0.8
    assert round(f1_score("tim burton", "tim burton filmmaker"), 3) == 0.8


def test_f1_no_overlap_is_zero():
    assert f1_score("cat", "dog") == 0.0


def test_support_recall():
    assert support_recall(["A", "B", "X"], ["A", "B"]) == 1.0
    assert support_recall(["A"], ["A", "B"]) == 0.5
    assert support_recall(["A"], []) == 1.0
