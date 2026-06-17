from hoprag.dataset import pool_chunks, examples_from_records

RECORDS = [
    {
        "id": "q1",
        "question": "Who directed the film Ed Wood?",
        "answer": "Tim Burton",
        "context": {
            "title": ["Ed Wood", "Tim Burton"],
            "sentences": [["Ed Wood is a 1994 film.", " It is a biopic."],
                          ["Tim Burton is an American filmmaker."]],
        },
        "supporting_facts": {"title": ["Ed Wood", "Tim Burton"], "sent_id": [0, 0]},
    },
    {
        "id": "q2",
        "question": "Where was Tim Burton born?",
        "answer": "Burbank",
        "context": {
            "title": ["Tim Burton", "Burbank"],
            "sentences": [["Tim Burton is an American filmmaker."],
                          ["Burbank is a city in California."]],
        },
        "supporting_facts": {"title": ["Tim Burton", "Burbank"], "sent_id": [0, 0]},
    },
]


def test_pool_dedupes_by_title_text():
    chunks = pool_chunks(RECORDS)
    titles = sorted(c.title for c in chunks)
    # "Tim Burton" appears in both records with identical text -> deduped to one
    assert titles == ["Burbank", "Ed Wood", "Tim Burton"]
    assert all(c.id for c in chunks)  # every chunk has a stable id


def test_examples_carry_gold():
    exs = examples_from_records(RECORDS)
    assert exs[0].question.startswith("Who directed")
    assert exs[0].answer == "Tim Burton"
    assert set(exs[0].gold_support_titles) == {"Ed Wood", "Tim Burton"}
