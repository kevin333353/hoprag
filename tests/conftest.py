import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest


class FakeRetriever:
    """Returns scripted chunks per query; records calls."""
    def __init__(self, script):
        # script: dict[query_substring] -> list[Chunk]
        self.script = script
        self.calls = []

    def search(self, query, k):
        self.calls.append(query)
        for key, chunks in self.script.items():
            if key in query:
                return chunks[:k]
        return []


class FakeClaude:
    """Returns scripted JSON dicts in order; records prompts."""
    def __init__(self, json_responses=None, text_responses=None):
        self.json_responses = list(json_responses or [])
        self.text_responses = list(text_responses or [])
        self.prompts = []

    def complete_json(self, prompt, schema):
        self.prompts.append(prompt)
        return self.json_responses.pop(0)

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.text_responses.pop(0)


@pytest.fixture
def chunks():
    from hoprag.types import Chunk  # lazy: types.py is created in Task 1
    return [
        Chunk(id="c1", title="Scott Derrickson", text="Scott Derrickson is an American director."),
        Chunk(id="c2", title="Ed Wood", text="Ed Wood is a 1994 film directed by Tim Burton."),
        Chunk(id="c3", title="Tim Burton", text="Tim Burton is an American filmmaker born in 1958."),
    ]
