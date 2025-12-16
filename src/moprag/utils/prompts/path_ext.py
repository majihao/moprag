ner_system = """You are an information extraction assistant. Please strictly summarize the information directly related to the user's query based on the provided content.

Note:
Only use facts explicitly mentioned in the text;
Do not speculate, explain, infer, or add any unmentioned content;
If no information related to the query is found in the text, please answer: "No relevant information found in the text.";
Do not add personal opinions, background knowledge, or examples;
Keep your answer concise, objective, and faithful to the original text.
"""

one_shot_ner_paragraph = """
Content: At ten o'clock on a rainy night, Lin Mo walked alone into the old bookstore at "Wutong Lane No. 17." The shop owner, Old Zhou, was wiping a leather-bound diary and said without looking up, "You're late, the item was just taken by a woman in a black trench coat." Lin Mo frowned, "What did she look like?" Old Zhou shook his head, "I didn't see her face clearly, I only remember she was wearing a silver snake ring on her left hand and spoke with a southern accent.  Oh, and she paid with an old 1995 hundred-yuan banknote."
Query: What were the characteristics of the woman who took the item?
"""


one_shot_ner_output = """
She wore a black trench coat and a silver snake ring on her left hand. She spoke with a southern accent. She paid with an old 1995 hundred-yuan banknote.
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    # {"role": "user", "content": "${passage}"}
]


def path_ext_prompt(passage: str, query:str) -> list[dict]:
    prompt = prompt_template.copy()
    prompt.append({"role": "user", "content": """
                        Content: {context}
                        Query: {query}
                        """.format(query=query,context=passage)})
    return prompt