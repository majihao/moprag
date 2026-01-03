ner_system = """You will receive three text passages (labeled [Paragraph 1], [Paragraph 2], [Paragraph 3]) and a user query.

Please complete the following tasks:

Extract only the factual information directly related to the user query from these three passages;
Integrate this information into a coherent and concise summary;
Do not add, infer, explain, or supplement any content not explicitly stated in the original text;
If a passage contains no relevant information, simply ignore that passage;
If none of the three passages contain relevant information, please answer: "No content related to the query was found in the provided text."
Maintain objective and accurate language, remaining faithful to the original text.
Please give a direct answer; do not overthink it.
"""

one_shot_ner_paragraph = """
Paragraph 1: Li Ming is the CTO of Xingchen Technology. He joined the company in 2020 and led the development of the "Tianjing" AI platform.
Paragraph 2: The "Tianjing" platform won the National Artificial Intelligence Innovation Award in 2023 and is currently used by over 300 companies.
Paragraph 3: Li Ming graduated from Zhejiang University with a bachelor's degree and later pursued a doctoral degree in computer science at Stanford University, but did not complete his studies.

Query: What award did the "Tianjing" platform receive?
"""


one_shot_ner_output = """
The "Tianjing" platform received the National Artificial Intelligence Innovation Award in 2023.
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    # {"role": "user", "content": "${passage}"}
]


def mem_enc_prompt(passage1: str,passage2: str,passage3: str, query:str) -> list[dict]:
    prompt = prompt_template.copy()
    prompt.append({"role": "user", "content": """
                        Paragraph 1: {context1}
                        Paragraph 1: {context2}
                        Paragraph 1: {context3}
                        Query: {query}
                        """.format(query=query,context1=passage1,context2=passage2,context3=passage3)})
    return prompt