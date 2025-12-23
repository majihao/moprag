

ner_system="""You are a professional knowledge graph construction expert, skilled at extracting entities and summary from story text.

Summarize this paragraph.

Note that: 
1. Strictly follow the format shown in the examples below, using "Input:" for the paragraph and "Entities:" and "Summary:" for the output.
"""


one_shot_ner_paragraph = """
Paragraph: In the enchanted forest of Elderglen, the young sorceress Lyra Moonshadow recovered the stolen Crystal of Aether from the lair of the Shadow Drake. With the help of her talking fox companion, Ember, she returned it to the Oracle Tree, restoring balance to the magical realm.
"""

one_shot_ner_output = """
Sorceress Lyra Moonshadow retrieved the stolen Crystal of Aether from the Shadow Drake’s lair and restored balance to the magical realm with her fox companion Ember.
"""

one_shot_ner_paragraph2="""
Paragraph: In the latest episode of the popular series "Galactic Adventures," Captain Lila Myles and her crew, Dr. Elena Torres and Pilot Jake Chen, embarked on a mission to the distant planet Zephyr. Their goal was to retrieve a rare mineral known as Zentar, essential for their starship's power.
"""

one_shot_ner_output2="""
Captain Lila Myles and her crew journeyed to planet Zephyr to collect the rare mineral Zentar in the latest "Galactic Adventures" episode.
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": one_shot_ner_paragraph2},
    {"role": "assistant", "content": one_shot_ner_output2}

]


def story_summary_prompt(passage)->list:
    
    prompt = prompt_template.copy()
    prompt.append({"role": "user", "content": """
                        Paragraph: {context}
                        """.format(context=passage)
                    })
    return prompt





