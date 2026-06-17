from hoprag import prompts_zh
from hoprag.naive_rag import NaiveRAG
from hoprag.types import Chunk


def test_zh_prompts_are_chinese_and_keep_schemas():
    c = [Chunk(id="c1", title="doc p.1", text="人工智慧基礎")]
    p = prompts_zh.synthesize_prompt("什麼是人工智慧?", c)
    assert "繁體中文" in p
    assert "c1" in p  # chunk id embedded for citation
    for s in (prompts_zh.DECOMPOSE_SCHEMA, prompts_zh.HOP_SCHEMA,
              prompts_zh.SYNTH_SCHEMA, prompts_zh.VERIFY_SCHEMA):
        assert s["type"] == "object" and "required" in s


def test_pipeline_uses_injected_chinese_prompts():
    from conftest import FakeRetriever, FakeClaude
    retr = FakeRetriever({"q": [Chunk(id="c1", title="t", text="x")]})
    claude = FakeClaude(json_responses=[{"answer": "答案", "cited_chunk_ids": ["c1"]}])
    NaiveRAG(retr, claude, prompts_mod=prompts_zh).answer("q")
    assert "繁體中文" in claude.prompts[0]  # the injected zh synthesize prompt was used
