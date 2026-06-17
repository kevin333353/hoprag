import hashlib
from dataclasses import dataclass

from hoprag.types import Chunk


@dataclass
class Example:
    qid: str
    question: str
    answer: str
    gold_support_titles: list[str]


def _chunk_id(title: str, text: str) -> str:
    h = hashlib.sha1(f"{title}::{text}".encode("utf-8")).hexdigest()[:12]
    return f"ch_{h}"


def pool_chunks(records: list[dict]) -> list[Chunk]:
    """Pool all context paragraphs across records into deduped Chunks."""
    seen: dict[str, Chunk] = {}
    for rec in records:
        titles = rec["context"]["title"]
        sentences = rec["context"]["sentences"]
        for title, sents in zip(titles, sentences):
            text = " ".join(s.strip() for s in sents).strip()
            cid = _chunk_id(title, text)
            if cid not in seen:
                seen[cid] = Chunk(id=cid, title=title, text=text, source_qid=rec["id"])
    return list(seen.values())


def examples_from_records(records: list[dict]) -> list[Example]:
    return [
        Example(
            qid=rec["id"],
            question=rec["question"],
            answer=rec["answer"],
            gold_support_titles=list(dict.fromkeys(rec["supporting_facts"]["title"])),
        )
        for rec in records
    ]


def load_hotpotqa(split: str = "validation", n: int = 300) -> list[dict]:
    """Thin HF loader (network). Not unit-tested; exercised by scripts/run_eval.py."""
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    return [ds[i] for i in range(min(n, len(ds)))]
