ner_system = """Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
"""

one_shot_ner_paragraph = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""


one_shot_ner_output = """Entities: Radio City, India, 3 July 2001, Hindi, English, May 2008, PlanetRadiocity.com
"""

one_shot_ner_paragraph2="""In the enchanted forest of Elderglen, the young sorceress Lyra Moonshadow recovered the stolen Crystal of Aether from the lair of the Shadow Drake. With the help of her talking fox companion, Ember, she returned it to the Oracle Tree, restoring balance to the magical realm."""


one_shot_ner_output2="""Entities: Elderglen, Lyra Moonshadow, Crystal of Aether, Shadow Drake, Ember, Oracle Tree, Magical realm"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": one_shot_ner_paragraph2},
    {"role": "assistant", "content": one_shot_ner_output2},
    # {"role": "user", "content": "${passage}"}
]


def ner_prompt_builder(passage: str) -> list[dict]:
    prompt = prompt_template.copy()
    prompt.append({"role": "user", "content": passage})
    return prompt