"""A small, self-contained corpus for the interactive demo.

The documents form short entity chains (film -> director/actor -> birthplace) so that
multi-hop questions have clear answers and run instantly without downloading HotpotQA.
"""

from hoprag.types import Chunk

DEMO_DOCS = [
    {"title": "Ed Wood (film)",
     "text": "Ed Wood is a 1994 American biographical film directed by Tim Burton. "
             "It stars Johnny Depp as the cult filmmaker Edward D. Wood Jr."},
    {"title": "Tim Burton",
     "text": "Tim Burton is an American filmmaker born on August 25, 1958, in Burbank, "
             "California. He is known for gothic fantasy films such as Edward Scissorhands."},
    {"title": "Burbank, California",
     "text": "Burbank is a city in Los Angeles County, California, often called the "
             "Media Capital of the World because many film and television studios are based there."},
    {"title": "Johnny Depp",
     "text": "Johnny Depp is an American actor born on June 9, 1963, in Owensboro, "
             "Kentucky. He has frequently collaborated with the director Tim Burton."},
    {"title": "Owensboro, Kentucky",
     "text": "Owensboro is a city in Kentucky located on the Ohio River. "
             "It is the fourth-largest city in the state."},
    {"title": "Inception (film)",
     "text": "Inception is a 2010 science fiction action film written and directed by "
             "Christopher Nolan, following a thief who steals secrets through dream-sharing."},
    {"title": "Christopher Nolan",
     "text": "Christopher Nolan is a British-American film director born on July 30, 1970, "
             "in London, England. His films include Inception, Interstellar and The Dark Knight."},
    {"title": "London",
     "text": "London is the capital and largest city of England and the United Kingdom, "
             "standing on the River Thames in the south-east of Great Britain."},
    {"title": "The Dark Knight (film)",
     "text": "The Dark Knight is a 2008 superhero film directed by Christopher Nolan. "
             "It is the second installment of his Batman trilogy and stars Christian Bale."},
    {"title": "Pulp Fiction",
     "text": "Pulp Fiction is a 1994 American crime film directed by Quentin Tarantino, "
             "known for its nonlinear storyline and eclectic dialogue."},
    {"title": "Quentin Tarantino",
     "text": "Quentin Tarantino is an American filmmaker born on March 27, 1963, in "
             "Knoxville, Tennessee. He is known for stylized violence and sharp dialogue."},
    {"title": "Knoxville, Tennessee",
     "text": "Knoxville is a city in eastern Tennessee on the Tennessee River. "
             "It is home to the University of Tennessee."},
    {"title": "Christian Bale",
     "text": "Christian Bale is an English actor born on January 30, 1974, in "
             "Haverfordwest, Wales. He played Batman in Christopher Nolan's Dark Knight trilogy."},
    {"title": "Haverfordwest",
     "text": "Haverfordwest is the county town of Pembrokeshire, Wales, "
             "situated on the Western Cleddau river."},
]

EXAMPLE_QUESTIONS = [
    {"question": "In which city was the director of Ed Wood born?", "answer": "Burbank"},
    {"question": "Where was the director of Inception born?", "answer": "London"},
    {"question": "In what city was the director of Pulp Fiction born?", "answer": "Knoxville"},
    {"question": "Where was the lead actor of Ed Wood born?", "answer": "Owensboro"},
    {"question": "In which town was the actor who played Batman in The Dark Knight born?",
     "answer": "Haverfordwest"},
]


def demo_chunks() -> list[Chunk]:
    return [Chunk(id=f"d{i}", title=d["title"], text=d["text"])
            for i, d in enumerate(DEMO_DOCS)]
