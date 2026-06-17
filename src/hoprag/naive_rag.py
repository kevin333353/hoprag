from hoprag import prompts
from hoprag.types import Result


class NaiveRAG:
    def __init__(self, retriever, claude, top_k: int = 5):
        self.retriever = retriever
        self.claude = claude
        self.top_k = top_k

    def answer(self, question: str) -> Result:
        chunks = self.retriever.search(question, self.top_k)
        synth = self.claude.complete_json(
            prompts.synthesize_prompt(question, chunks), prompts.SYNTH_SCHEMA
        )
        return Result(
            question=question,
            answer=synth["answer"],
            cited_chunk_ids=synth["cited_chunk_ids"],
            n_claude_calls=1,
            n_retrievals=1,
        )
