"""Traditional-Chinese prompt builders for the RAG tool.

Same function names and JSON schemas as `hoprag.prompts` (so it can be injected into
NaiveRAG/AgenticRAG via the `prompts_mod` parameter), but the instructions are in
Chinese and ask for Traditional-Chinese answers — for querying Chinese documents.
"""

from hoprag.prompts import (
    _format_chunks,
    DECOMPOSE_SCHEMA, HOP_SCHEMA, SYNTH_SCHEMA, VERIFY_SCHEMA,
)

__all__ = [
    "decompose_prompt", "hop_prompt", "synthesize_prompt", "verify_prompt",
    "DECOMPOSE_SCHEMA", "HOP_SCHEMA", "SYNTH_SCHEMA", "VERIFY_SCHEMA",
]


def decompose_prompt(question: str) -> str:
    return (
        f"你正在規劃一次多跳檢索，用來回答以下問題：\n{question}\n\n"
        "請把它拆成有順序的子問題，並給出最適合的「第一個檢索查詢字串」。"
        '只用 JSON 回覆：{"subquestions": ["..."], "first_query": "..."}'
    )


def hop_prompt(question: str, chunks) -> str:
    return (
        f"原始問題：{question}\n\n"
        f"目前已蒐集到的證據：\n{_format_chunks(chunks)}\n\n"
        "判斷這些證據是否足以回答原始問題。若不足，給出下一個檢索查詢字串"
        "（針對還缺的資訊，善用你剛從證據裡學到的名詞或實體）。"
        '只用 JSON 回覆：{"reasoning": "...", "sufficient": true或false, "next_query": "..."}'
    )


def synthesize_prompt(question: str, chunks) -> str:
    return (
        f"問題：{question}\n\n"
        f"證據：\n{_format_chunks(chunks)}\n\n"
        "請只根據上述證據，用繁體中文盡量精簡、準確地回答問題；"
        "若證據不足以回答，請明說「證據不足」。並標註你用到的 chunk id。"
        '只用 JSON 回覆：{"answer": "...", "cited_chunk_ids": ["c1", "..."]}'
    )


def verify_prompt(question: str, answer: str, chunks) -> str:
    return (
        f"問題：{question}\n建議答案：{answer}\n\n"
        f"證據：\n{_format_chunks(chunks)}\n\n"
        "檢查這個建議答案是否完全被引用的證據支持。若否，請改寫成證據真正支持的內容，"
        "並修正引用。答案請用繁體中文。"
        '只用 JSON 回覆：{"supported": true或false, "revised_answer": "...", '
        '"cited_chunk_ids": ["c1", "..."]}'
    )
