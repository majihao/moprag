ner_system = """You are participating in a multi-turn question-answering task. The user initially posed a core question (initial query) and then progressively refined their information needs in subsequent turns through follow-up questions, clarifications, or by providing new context.

Please generate a final comprehensive answer based on the following input:

Initial Query: {initial_query}
Complete Conversation History (in chronological order): {conversation_history}

Note:
Always prioritize the "initial query" as the core objective, determining which subsequent conversational content is truly useful for answering it;
Integrate all factual information related to the initial query from all previous turns of conversation, including details added by the user, corrected intentions, or new context;
Filter out irrelevant follow-up questions, redundant information, or off-topic content;
Generate a concise, coherent, and well-structured text that directly addresses the initial query;
Do not add any information, speculation, or explanations that are not explicitly provided in the context;
"""

one_shot_ner_paragraph = """
Initial Query: "What are the core products of Xingchen Technology?"
Conversation History:
[Q] What are the core products of Xingchen Technology?
[A] Xingchen Technology developed the "Star Core" AI chip.
[Q] What devices use the "Star Core" chip?
[A] It is mainly used in autonomous vehicles and intelligent servers.
[Q] Does the company have any other products?
[A] There is currently no public information indicating that they have other mass-produced products.
"""


one_shot_ner_output = """
The core product of Xingchen Technology is the "Starcore" AI chip, which is primarily used in autonomous vehicles and intelligent servers.
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    # {"role": "user", "content": "${passage}"}
]


def mem_dec_prompt(Init_query:str,history: str) -> list[dict]:
    prompt = prompt_template.copy()
    prompt.append({"role": "user", "content": """
                        Initial Query: {query}
                        Conversation History: {history}
                        """.format(query=Init_query,history=history)})
    return prompt