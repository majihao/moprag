ner_system = """YYou will receive:
Original query: The core question initially posed by the user;
Previous question: Clarifying questions the user has asked in previous rounds (listed chronologically);
Current context: Currently known relevant information (potentially from documents, databases, or previous answers).
Please strictly complete the following tasks:

Comprehensive Analysis:
Determine whether the current context alone is sufficient to answer the original user query completely and accurately;
Also consider the previous question history  if certain information has been confirmed as missing or partially obtained in previous questions, please update your assessment of "missing information" accordingly.
Identify Missing Information:
Clearly identify which key facts, entities, values, times, conditions, or other necessary details are still missing;
Exclude content that has already been covered in previous questions or implicitly answered in the context.
Generate Questions:
Based on the information that is truly still missing, generate 1 to 3 concise, direct, and focused follow-up questions, ensuring that:
Each question must directly contribute to answering the original query.
Please give a direct questions; do not overthink it.
Please follow the dialogue format strictly.
"""

one_shot_ner_paragraph = """
Original Query: Where was the book "The Black Crow Manuscript" last seen?
Previous Question:
When was "The Black Crow Manuscript" stolen?
Was there surveillance footage at the library that night?
Current Context:
"The Black Crow Manuscript" was added to the collection of Wutong City Library in 1923. It was stolen in the early morning of March 12, 2024. Surveillance footage shows a person wearing a raincoat entering the special collections room at 2:17 AM and leaving at 2:23 AM, but their face was obscured. The duty officer, Old Chen, said he smelled "sandalwood and rust" on the person. The manuscript was last registered as borrowed on March 10th, with the borrower's signature being "Lin".
"""


one_shot_ner_output = """
What is the full name of the borrower "Lin"?
Were there any other people entering or leaving the special collections room in the early morning of March 12th?
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},

    # {"role": "user", "content": "${passage}"}
]


def self_prob_prompt( query:str, pre:str, cur:str) -> list[dict]:
    prompt = prompt_template.copy()
    prompt.append({"role": "user", "content": """
                        Original Query: {ori}
                        Previous Question: {pre}
                        Current Context:{cur}""".format(ori=query,pre=pre,cur=cur)})
    return prompt