import json
import pytest
from hoprag.claude_client import ClaudeClient, ClaudeError

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class StubClient(ClaudeClient):
    def __init__(self, raw_outputs):
        super().__init__()
        self._raw_outputs = list(raw_outputs)
        self.calls = 0

    def _run_raw(self, prompt: str) -> str:
        self.calls += 1
        return self._raw_outputs.pop(0)


def _wire(text):
    return json.dumps({"result": text, "usage": {}, "total_cost_usd": 0.0})


def test_complete_returns_result_text():
    c = StubClient([_wire("hello world")])
    assert c.complete("hi") == "hello world"


def test_complete_json_parses_valid():
    c = StubClient([_wire('{"answer": "42"}')])
    assert c.complete_json("q", SCHEMA) == {"answer": "42"}


def test_complete_json_retries_then_succeeds():
    c = StubClient([_wire("not json"), _wire('{"answer": "ok"}')])
    assert c.complete_json("q", SCHEMA, max_retries=2) == {"answer": "ok"}
    assert c.calls == 2


def test_complete_json_raises_after_retries():
    c = StubClient([_wire("nope"), _wire("still nope")])
    with pytest.raises(ClaudeError):
        c.complete_json("q", SCHEMA, max_retries=2)
