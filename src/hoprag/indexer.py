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
    # mode="overwrite" replaces any existing table — robust to re-runs over a
    # persistent db_path (the prior list_tables() existence check did not fire).
    return db.create_table(table_name, data=rows, mode="overwrite")


def make_bge_embed_fn(model_name: str = "BAAI/bge-small-en-v1.5"):
    # Use the OS (Windows) trust store so model downloads work behind a corporate
    # TLS-inspecting proxy (self-signed CA in the chain). Harmless elsewhere.
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts):
        return model.encode(list(texts), normalize_embeddings=True).tolist()

    return embed
