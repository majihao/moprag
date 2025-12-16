
prompt_template = [
                    {"role": "system", "content": """You are an expert in information extraction. Extract all explicit (subject, predicate, object) triples about the given Entities from the following sentence
                        
                        Requirements:
                        Replace pronouns (e.g., "he", "she", "it", etc.) with the appropriate entity name. 
                        Extract triples that directly involve the provided entities. 
                        Don't analyze too much, extract the triplets related to entities directly
                        Maintain the accuracy of meaning and context from the source text.
                        """},
                    {"role": "user", "content": """
                        context : "Albert Einstein proposed the theory of relativity. He also made significant contributions to physics."
                        Entities: ["Albert Einstein", "the theory of relativity", "physics"]
                        """},
                    {"role": "assistant", "content": """
                        ["Albert Einstein", "proposed", "the theory of relativity"],
                        ["Albert Einstein", "made", "significant contributions to physics")]
                        """},
                    {"role": "user", "content": """
                        context : Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.
                        Entities:  ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
                        """},
                    {"role": "assistant", "content": """
                        ["Radio City", "located in", "India"],
                        ["Radio City", "is", "private FM radio station"],
                        ["Radio City", "started on", "3 July 2001"],
                        ["Radio City", "plays songs in", "Hindi"],
                        ["Radio City", "plays songs in", "English"],
                        ["Radio City", "forayed into", "New Media"],
                        ["Radio City", "launched", "PlanetRadiocity.com"],
                        ["PlanetRadiocity.com", "launched in", "May 2008"],
                        ["PlanetRadiocity.com", "is", "music portal"],
                        ["PlanetRadiocity.com", "offers", "news"],
                        ["PlanetRadiocity.com", "offers", "videos"],
                        ["PlanetRadiocity.com", "offers", "songs"]
                        """}
                ]


def teiple_prompt_builder(passage: str, entities:list[str]) -> list[dict]:
    prompt = prompt_template.copy()
    prompt.append({"role": "user", "content": """
                        context: {context}
                        Entities: {entities}
                        """.format(entities=entities,context=passage)
                    },)
    return prompt