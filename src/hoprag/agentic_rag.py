from dataclasses import dataclass

from hoprag import prompts
from hoprag.types import HopStep, Result


@dataclass
class AgenticConfig:
    decompose: bool = True
    sufficiency_check: bool = True
    verify_citations: bool = True
    max_hops: int = 5
    fixed_hops: int = 2          # used when sufficiency_check is False
    top_k: int = 5

    def __post_init__(self):
        if self.fixed_hops > self.max_hops:
            raise ValueError(
                f"fixed_hops ({self.fixed_hops}) must be <= max_hops ({self.max_hops})"
            )


def _dedup(chunks):
    seen, out = set(), []
    for c in chunks:
        if c.id not in seen:
            seen.add(c.id)
            out.append(c)
    return out


class AgenticRAG:
    def __init__(self, retriever, claude, config: AgenticConfig | None = None):
        self.retriever = retriever
        self.claude = claude
        self.config = config or AgenticConfig()

    def answer(self, question: str) -> Result:
        cfg = self.config
        res = Result(question=question, answer="", cited_chunk_ids=[])

        # (1) decompose / plan
        if cfg.decompose:
            plan = self.claude.complete_json(
                prompts.decompose_prompt(question), prompts.DECOMPOSE_SCHEMA)
            res.n_claude_calls += 1
            current_query = plan["first_query"]
        else:
            current_query = question

        gathered = []
        for hop in range(cfg.max_hops):
            chunks = self.retriever.search(current_query, cfg.top_k)
            res.n_retrievals += 1
            gathered = _dedup(gathered + chunks)

            step = self.claude.complete_json(
                prompts.hop_prompt(question, gathered), prompts.HOP_SCHEMA)
            res.n_claude_calls += 1
            sufficient = bool(step["sufficient"])
            res.trace.append(HopStep(
                query=current_query,
                retrieved_ids=[c.id for c in chunks],
                reasoning=step["reasoning"],
                sufficient=sufficient,
            ))

            if cfg.sufficiency_check:
                done = sufficient
            else:
                done = (hop + 1) >= cfg.fixed_hops
            if done:
                break
            current_query = step["next_query"] or current_query

        # (4) synthesize with citations
        synth = self.claude.complete_json(
            prompts.synthesize_prompt(question, gathered), prompts.SYNTH_SCHEMA)
        res.n_claude_calls += 1
        res.answer = synth["answer"]
        res.cited_chunk_ids = synth["cited_chunk_ids"]

        # (5) verify citations
        if cfg.verify_citations:
            ver = self.claude.complete_json(
                prompts.verify_prompt(question, res.answer, gathered),
                prompts.VERIFY_SCHEMA)
            res.n_claude_calls += 1
            if not ver["supported"]:
                res.answer = ver["revised_answer"]
                res.cited_chunk_ids = ver["cited_chunk_ids"]

        return res
