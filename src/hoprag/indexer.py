import lancedb


def build_index(chunks, embed_fn, db_path: str, table_name: str = "chunks"):
    """Embed chunk texts and store in a LanceDB table. Returns the opened table."""
    vectors = embed_fn([c.text for c in chunks])
    rows = [
        {"id": c.id, "title": c.title, "text": c.text,
         "source_qid": c.source_qid or "", "vector": v}
        for c, v in zip(chunks, vectors)
    ]
    db = lancedb.connect(db_path)
    if table_name in db.list_tables():
        db.drop_table(table_name)
    return db.create_table(table_name, data=rows)


def make_bge_embed_fn(model_name: str = "BAAI/bge-small-en-v1.5"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts):
        return model.encode(list(texts), normalize_embeddings=True).tolist()

    return embed
