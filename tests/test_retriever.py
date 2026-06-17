from hoprag.indexer import build_index
from hoprag.retriever import Retriever
from hoprag.types import Chunk

VOCAB = ["burton", "wood", "burbank", "film", "city"]


def fake_embed(texts):
    out = []
    for t in texts:
        low = t.lower()
        out.append([float(low.count(w)) for w in VOCAB])
    return out


def test_retriever_returns_nearest_chunk(tmp_path):
    chunks = [
        Chunk(id="c1", title="Ed Wood", text="Ed Wood is a film"),
        Chunk(id="c2", title="Burbank", text="Burbank is a city"),
        Chunk(id="c3", title="Tim Burton", text="Tim Burton directed the film"),
    ]
    index = build_index(chunks, embed_fn=fake_embed, db_path=str(tmp_path / "db"))
    r = Retriever(index, embed_fn=fake_embed)
    top = r.search("which city", k=1)
    assert top[0].id == "c2"
    top2 = r.search("burton film", k=2)
    assert {c.id for c in top2} == {"c1", "c3"} or top2[0].id == "c3"


def test_build_index_overwrites_existing(tmp_path):
    db = str(tmp_path / "db")
    a = [Chunk(id="c1", title="A", text="city")]
    build_index(a, embed_fn=fake_embed, db_path=db)
    # rebuilding over the same path must not raise "table already exists"
    b = [Chunk(id="c2", title="B", text="film")]
    table = build_index(b, embed_fn=fake_embed, db_path=db)
    ids = {row["id"] for row in table.search(fake_embed(["film"])[0]).limit(5).to_list()}
    assert ids == {"c2"}  # overwritten, old c1 gone
