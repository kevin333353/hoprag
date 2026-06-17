def _format_chunks(chunks) -> str:
    return "\n".join(f"[{c.id}] ({c.title}) {c.text}" for c in chunks)


DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "subquestions": {"type": "array", "items": {"type": "string"}},
        "first_query": {"type": "string"},
    },
    "required": ["first_query"],
}

HOP_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "sufficient": {"type": "boolean"},
        "next_query": {"type": "string"},
    },
    "required": ["reasoning", "sufficient", "next_query"],
}

SYNTH_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited_chunk_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "cited_chunk_ids"],
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "revised_answer": {"type": "string"},
        "cited_chunk_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["supported", "revised_answer", "cited_chunk_ids"],
}


def decompose_prompt(question: str) -> str:
    return (
        f"You are planning a multi-hop search to answer:\n{question}\n\n"
        "Break it into ordered subquestions and give the single best FIRST search query. "
        'Respond as JSON: {"subquestions": [...], "first_query": "..."}'
    )


def hop_prompt(question: str, chunks) -> str:
    return (
        f"Original question: {question}\n\n"
        f"Evidence gathered so far:\n{_format_chunks(chunks)}\n\n"
        "Decide if the evidence is sufficient to answer the original question. "
        "If not, give the next search query that targets the missing fact "
        "(use entities you just learned). "
        'Respond as JSON: {"reasoning": "...", "sufficient": true|false, "next_query": "..."}'
    )


def synthesize_prompt(question: str, chunks) -> str:
    return (
        f"Question: {question}\n\n"
        f"Evidence:\n{_format_chunks(chunks)}\n\n"
        "Answer the question as briefly as possible (a short span, like HotpotQA). "
        "Cite the chunk ids you used. "
        'Respond as JSON: {"answer": "...", "cited_chunk_ids": ["c1", ...]}'
    )


def verify_prompt(question: str, answer: str, chunks) -> str:
    return (
        f"Question: {question}\nProposed answer: {answer}\n\n"
        f"Evidence:\n{_format_chunks(chunks)}\n\n"
        "Check whether the proposed answer is fully supported by the cited evidence. "
        "If not, revise it to what the evidence supports and fix the citations. "
        'Respond as JSON: {"supported": true|false, "revised_answer": "...", '
        '"cited_chunk_ids": ["c1", ...]}'
    )
