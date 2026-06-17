from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    source_qid: str | None = None


@dataclass
class HopStep:
    query: str
    retrieved_ids: list[str]
    reasoning: str
    sufficient: bool


@dataclass
class Result:
    question: str
    answer: str
    cited_chunk_ids: list[str]
    trace: list[HopStep] = field(default_factory=list)
    n_claude_calls: int = 0
    n_retrievals: int = 0
