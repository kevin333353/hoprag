import pytest
from hoprag.ingest import window_text, chunk_pages


def test_window_text_size_and_count():
    w = window_text("0123456789" * 100, size=400, overlap=100)  # 1000 chars, step 300
    assert all(len(x) <= 400 for x in w)
    assert len(w) == 3  # [0:400], [300:700], [600:1000]


def test_window_text_short_and_empty():
    assert window_text("") == []
    assert window_text("   ") == []
    assert window_text("hello") == ["hello"]


def test_window_text_rejects_bad_params():
    with pytest.raises(ValueError):
        window_text("x", size=50, overlap=50)


def test_chunk_pages_titles_ids_source():
    cs = chunk_pages([(1, "x" * 600), (2, "y" * 100)], "doc1", size=500, overlap=50)
    assert any(c.title == "doc1 p.1" for c in cs)
    assert any(c.title == "doc1 p.2" for c in cs)
    assert all(c.source_qid == "doc1" for c in cs)
    assert len({c.id for c in cs}) == len(cs)  # ids unique
