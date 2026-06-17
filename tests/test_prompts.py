from hoprag import prompts
from hoprag.types import Chunk

CHUNKS = [Chunk(id="c1", title="Tim Burton", text="American filmmaker")]


def test_hop_prompt_includes_question_and_chunk_ids():
    p = prompts.hop_prompt("who directed Ed Wood?", CHUNKS)
    assert "who directed Ed Wood?" in p
    assert "c1" in p and "Tim Burton" in p


def test_schemas_are_objects():
    for s in (prompts.DECOMPOSE_SCHEMA, prompts.HOP_SCHEMA,
              prompts.SYNTH_SCHEMA, prompts.VERIFY_SCHEMA):
        assert s["type"] == "object"
        assert "required" in s
