from hoprag import prompts as _default_prompts
from hoprag.types import HopStep, Result


class NaiveRAG:
    def __init__(self, retriever, claude, top_k: int = 5, prompts_mod=None):
        self.retriever = retriever
        self.claude = claude
        self.top_k = top_k
        self.prompts = prompts_mod or _default_prompts  # inject prompts_zh for Chinese

    def answer(self, question: str) -> Result:
        chunks = self.retriever.search(question, self.top_k)
        synth = self.claude.complete_json(
            self.prompts.synthesize_prompt(question, chunks), self.prompts.SYNTH_SCHEMA
        )
        return Result(
            question=question,
            answer=synth["answer"],
            cited_chunk_ids=synth.get("cited_chunk_ids") or [],
            trace=[HopStep(
                query=question,
                retrieved_ids=[c.id for c in chunks],
                reasoning="naive single-shot retrieval",
                sufficient=True,
            )],
            n_claude_calls=1,
            n_retrievals=1,
        )
