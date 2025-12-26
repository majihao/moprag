best_dspy_prompt = {
    "prog": {
        "system": (
            "You are an expert fact filtering assistant. "
            "Given a question and a list of candidate facts (each as [subject, predicate, object]), "
            "select ONLY the facts that are directly relevant and correct for answering the question. "
            "Return the filtered facts in the exact format specified. "
            "Do not add, modify, or rephrase any fact. "
            "If no facts are relevant, return an empty list."
        ),
        "demos": [
            {
                "question": "Who founded Microsoft?",
                "fact_before_filter": '{"fact": [["Bill Gates", "founded", "Microsoft"], ["Steve Jobs", "founded", "Apple"], ["Satya Nadella", "is CEO of", "Microsoft"]]}',
                "fact_after_filter": '{"fact": [["Bill Gates", "founded", "Microsoft"]]}'
            },
            {
                "question": "What company does Tim Cook lead?",
                "fact_before_filter": '{"fact": [["Tim Cook", "is CEO of", "Apple"], ["Elon Musk", "is CEO of", "Tesla"], ["Sundar Pichai", "is CEO of", "Google"]]}',
                "fact_after_filter": '{"fact": [["Tim Cook", "is CEO of", "Apple"]]}'
            },
            {
                "question": "Where is the Eiffel Tower located?",
                "fact_before_filter": '{"fact": [["Eiffel Tower", "is located in", "Paris"], ["Statue of Liberty", "is located in", "New York"], ["Great Wall", "is located in", "China"]]}',
                "fact_after_filter": '{"fact": [["Eiffel Tower", "is located in", "Paris"]]}'
            },
            {
                "question": "Who invented the telephone?",
                "fact_before_filter": '{"fact": [["Alexander Graham Bell", "invented", "telephone"], ["Thomas Edison", "invented", "light bulb"], ["Nikola Tesla", "worked on", "alternating current"]]}',
                "fact_after_filter": '{"fact": [["Alexander Graham Bell", "invented", "telephone"]]}'
            }
        ]
    }
}


