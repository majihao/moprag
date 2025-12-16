ner_system="""You are an expert in factual information extraction. Given a list of named entities and a context passage, generate a concise behavioral summary for each entity based strictly on the provided text.
Instructions:
- Only use information explicitly stated or directly implied in the context.
- Do not use external knowledge, assumptions, or inference beyond what is clearly supported.
- If an entity is mentioned but no specific action, role, or state is described, output: "Not explicitly described."
- Each summary must be a single, clear, grammatical sentence.
- Preserve the exact entity name as given in the input list.
- Output a valid JSON object where each key is an entity string and each value is its summary."""


one_shot_ner_paragraph = """
context = "In Q2 2024, Alice, the CTO of TechNova Inc., led the launch of a new cloud platform. Bob, an industry analyst, later criticized the platform's security design in a public report.
entities = ["Alice", "Bob", "TechNova Inc."]
"""

one_shot_ner_output = """{
  "Alice": "As CTO of TechNova Inc., led the launch of a new cloud platform.",
  "Bob": "Criticized the platform's security design in a public report.",
  "TechNova Inc.": "Launched a new cloud platform led by its CTO Alice."
}"""

one_shot_ner_paragraph2="""
context = "In 2023, Dr. Elena Martinez, a lead researcher at the Global Health Initiative (GHI), published a breakthrough study showing a 70 reduction in Malaria transmission using a new vaccine regimen in sub-Saharan Africa."
entities = ["Dr. Elena Martinez", "Global Health Initiative (GHI)", "Malaria"]
"""

one_shot_ner_output2="""
{
  "Dr. Elena Martinez": "Published a breakthrough study showing a 70 reduction in Malaria transmission using a new vaccine regimen.",
  "Global Health Initiative (GHI)": "Employed Dr. Elena Martinez as a lead researcher who published a breakthrough study on Malaria reduction.",
  "Malaria": "Showed a 70 reduction in transmission due to a new vaccine regimen in a study published by Dr. Elena Martinez."
}
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": one_shot_ner_paragraph2},
    {"role": "assistant", "content": one_shot_ner_output2}

]


def entities_summary_prompt(passage,entities)->list:
    
    prompt = prompt_template.copy()
    prompt.append({"role": "user", "content": """
                        context: {context}
                        entities: {entities}
                        """.format(entities=entities,context=passage)
                    })
    return prompt

    pass



