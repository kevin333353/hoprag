import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """HotpotQA official normalization: lowercase, strip punctuation/articles/extra ws."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def support_recall(pred_titles: list[str], gold_titles: list[str]) -> float:
    gold = set(gold_titles)
    if not gold:
        return 1.0
    return len(set(pred_titles) & gold) / len(gold)


def support_precision(pred_titles: list[str], gold_titles: list[str]) -> float:
    pred = set(pred_titles)
    if not pred:
        return 1.0
    gold = set(gold_titles)
    return len(pred & gold) / len(pred)
