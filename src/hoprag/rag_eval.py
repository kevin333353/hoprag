"""Reference-free RAG quality scoring — the RAG Triad (TruLens / RAGAS-style),
computed via LLM-as-judge. No ground-truth answer required.

Three 0-100 scores:
  - context_relevance: are the retrieved chunks relevant to the question?
  - groundedness:      is the answer supported by the retrieved chunks (vs hallucinated)?
  - answer_relevance:  does the answer actually address the question?
"""

TRIAD_SCHEMA = {
    "type": "object",
    "properties": {
        "context_relevance": {"type": "number"},
        "groundedness": {"type": "number"},
        "answer_relevance": {"type": "number"},
        "rationale": {"type": ["string", "null"]},
    },
    "required": ["context_relevance", "groundedness", "answer_relevance"],
}


def triad_prompt(question: str, answer: str, contexts: list[str]) -> str:
    ctx = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)) or "(沒有檢索到任何內容)"
    return (
        "你是一個 RAG（檢索增強生成）系統的嚴格評審。針對下面的（問題、檢索到的內容、答案），"
        "用 RAG Triad 的三個面向各打 0–100 分（整數）：\n"
        "1) context_relevance：檢索到的內容整體跟「問題」的相關程度。\n"
        "2) groundedness：答案的內容有多少是「被檢索內容支持」的；憑空捏造（幻覺）越多分數越低。\n"
        "3) answer_relevance：答案有多直接、完整地回答了「問題」。\n\n"
        f"問題：{question}\n\n檢索到的內容：\n{ctx}\n\n答案：{answer}\n\n"
        '只用 JSON 回覆：{"context_relevance":0-100,"groundedness":0-100,'
        '"answer_relevance":0-100,"rationale":"一句話理由"}'
    )


def _clamp(x) -> float:
    try:
        return max(0.0, min(100.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def score_triad(claude, question: str, answer: str, contexts: list[str]) -> dict:
    """Returns {context_relevance, groundedness, answer_relevance, rationale, overall}.
    Scores are clamped to [0, 100]; a fresh dict is returned (judge output not mutated)."""
    s = claude.complete_json(triad_prompt(question, answer, contexts), TRIAD_SCHEMA)
    cr, g, ar = _clamp(s["context_relevance"]), _clamp(s["groundedness"]), _clamp(s["answer_relevance"])
    return {
        "context_relevance": cr,
        "groundedness": g,
        "answer_relevance": ar,
        "rationale": s.get("rationale"),
        "overall": round((cr + g + ar) / 3, 1),
    }
