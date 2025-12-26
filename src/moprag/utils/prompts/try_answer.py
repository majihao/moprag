ner_system = """You will receive a user query and a piece of relevant contextual information.

Please strictly follow these rules:

If the contextual information contains enough information to directly and clearly answer the query, please output only the answer itself (do not explain, and do not add prefixes such as "The answer is:");
If the contextual information is missing, ambiguous, or does not support a definitive answer, please output only: No;
Do not fabricate, speculate, infer, or use external knowledge;
Do not output any other text, punctuation, spaces, or formatting.
Please give a direct answer; do not overthink it.
"""

one_shot_ner_paragraph = """
Context: Albert Einstein was born on March 14, 1879, in Ulm, Germany.
Query: In what year was Einstein born?
"""


one_shot_ner_output = """
1879
"""

one_shot_ner_paragraph1 = """
Context: Albert Einstein was born on March 14, 1879, in Ulm, Germany.
Query: In what year did Einstein die?
"""


one_shot_ner_output1 = """
No
"""

prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": one_shot_ner_paragraph1},
    {"role": "assistant", "content": one_shot_ner_output1},
    # {"role": "user", "content": "${passage}"}
]


def try_answer_prompt(passage: str, query:str) -> list[dict]:
    prompt = prompt_template.copy()
    prompt.append({"role": "user", "content": """
                        Content: {context}
                        Query: {query}
                        """.format(query=query,context=passage)})
    return prompt