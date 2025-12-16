
ner_system = """Can answer the query directly without information???
                   
              Note:
              Just answer "Yes" or "No", no need to provide the query answer.
              Please answer strictly according to the format(Answer: ??)
"""


one_shot_ner_paragraph=""" Query: Who I am？
                           Can you answer this question？
"""

one_shot_ner_output = """No"""



prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},


]


def judge_query(passage)->list:
    
    prompt = prompt_template.copy()
    prompt.append({"role": "user", "content": """
                        Query: {Query}
                        Can you answer this question？S
                        """.format(Query=passage)
                    })
    return prompt


