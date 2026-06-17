from hoprag.types import Chunk


class Retriever:
    def __init__(self, table, embed_fn):
        self.table = table
        self.embed_fn = embed_fn

    def search(self, query: str, k: int) -> list[Chunk]:
        qvec = self.embed_fn([query])[0]
        hits = self.table.search(qvec).limit(k).to_list()
        return [
            Chunk(id=h["id"], title=h["title"], text=h["text"],
                  source_qid=h["source_qid"] or None)
            for h in hits
        ]
