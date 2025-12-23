
import logging
import os
from abc import ABC, abstractmethod
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
# from .prompts.prompt_template_manager import PromptTemplateManager

from .prompts.judge_if_Answer import judge_query
from .prompts.mem_enc import mem_enc_prompt
from .prompts.path_ext import path_ext_prompt
from .prompts.try_answer import try_answer_prompt
from .prompts.self_prob import self_prob_prompt

from typing import Optional, List, Dict
import re



# from .config_utils import BaseConfig

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class PoolAgent():
    
    def __init__(self, model, client,max_completion_tokens,stop_sequence,temperature):
        self.client = client
        self.model_name=model

        self.max_completion_tokens=1500
        self.stop_sequence=None
        self.temperature=0
        self.top_p=1

    def _call_llm(self, messages, max_completion_tokens=1500, stop_sequence=None, temperature=0, top_p=1):
        
        try:
            # Call OpenAI API interface to generate summary
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                stop=stop_sequence,
                temperature=0,
                frequency_penalty=0,
                presence_penalty=0,
                top_p=1,
            )
            response=response.choices[0].message.content
            return response

        except Exception as e:
            
            logging.error(f"LLM call failed: {e}")
            return str(e)

    
    def djudgprocess(self, query):
         
        messages=judge_query(query)
        response = self._call_llm(messages=messages)
        response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        
        if "Yes" in  response:
            return True
        else:
            return False

    
    def path_ext(self, query ,inf):
         
        messages=path_ext_prompt(inf, query)
        response = self._call_llm(messages=messages)
        response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        
        return response

    
    def mem_enc(self, query ,inf1,inf2,inf3):
         
        messages=mem_enc_prompt(inf1,inf2,inf3, query)
        response = self._call_llm(messages=messages)
        response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        
        return response
    
    def try_answer(self, query ,inf):
         
        messages=try_answer_prompt(inf, query)
        response = self._call_llm(messages=messages)
        response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        
        return response

    def self_pro(self,query=None, pre=None, cur=None):
        
        query = query if query is not None else ""
        pre = pre if pre is not None else ""
        cur = cur if cur is not None else ""
        messages=self_prob_prompt(query, pre, cur)
        response = self._call_llm(messages=messages)
        response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)

        questions = re.split(r'(?<=\?)\s*', response)
        # 移除可能产生的空字符串
        questions = [q for q in questions if q] 
        return questions


    
    




    

    # def exprocess(self, query,context):
         
    #     messages=[{"role": "system", "content": """You are a helpful assistant."""},
    #               {"role": "user", "content": """You are an intelligent assistant that extracts only the information  relevant to the user's query from the provided context.
    #                                   Carefully read the user's query and the context below.
    #                                   Extract sentence or informeration that are  related to the query. Do not add explanations, or external knowledge.
                                      
    #                                   Note:
    #                                   If no relevant information exists, respond with: "No information"
    #                                   Please answer strictly according to the format(Information: ??)

    #                                   Query: {query}
                   
    #                                   Context: {context} 

    #                                   Information: ??""".format(query=query, context=context),
    #                 },
    #             ]
                
    #     response=self._call_llm(messages=messages)
        
    #     try:

    #         Information_match = re.search(r'Information:\s*(.*)', response, re.IGNORECASE)
          
    #         information = Information_match.group(1)  if Information_match else "No information"
            
    #     except Exception as e:

    #         information = "No information"
        
    #     return information

    # def aggregation_process(self, query,context_inf,summary_inf):
         
    #     messages=[{"role": "system", "content": """You are a helpful assistant."""},
    #               {"role": "user", "content": """
    #                                   Please summarize useful information about the Query from  Context1 and Context2
    #                                   Please answer strictly according to the format(**Summary of Useful Information**\n??)
                   
    #                                   Query: {query}
                                      
    #                                   Please integrate these useful information
    #                                   Context1: {context_inf} 
    #                                   Context2: {summary_inf}
                   
    #                                   **Summary of Useful Information**
    #                                   ??""".format(query=query, context_inf=context_inf,summary_inf=summary_inf),
    #                 },
    #             ]
                
    #     response=self._call_llm(messages=messages)
    #     try:
    #         Information_match = re.search(r'\*\*Summary of Useful Information\*\*\s*\n\s*(.+)', response)
    #         if Information_match:
    #             information = Information_match.group(1) if Information_match else "Non"
            
    #     except Exception as e:

    #         information = "No informeration"
        
    #     return information
        
   
    # def judgprocess(self, query,information):
         
    #     messages=[{"role": "system", "content": """You are a helpful assistant."""},
    #             {"role": "user","content": """Can the best answer be obtained from known information???
    #                                   Note:
    #                                   Just answer "Yes" or "No", no need to provide the query answer.
    #                                   Please answer strictly according to the format(Answer: ??)
                 
    #                                   Query: {query}

    #                                   Information: {context} 

    #                                   Can the answer be obtained from known information???
    #                                   Answer: ??""".format(query=query, context=information),
    #                 },
    #             ]
    #     response=self._call_llm(messages=messages)
     
    #     try:
    #         judge_match = re.search(r'Answer:\s*(.*)', response)
    #         if judge_match:
    #             judge_answer = judge_match.group(1)
    #         else:
    #             judge_answer = "No"
    #     except Exception as e:
    #             judge_answer = "No"
        
    #     if "Yes" in judge_answer:
    #         return True
    #     else:
    #         return False

    
    
        
        
    def danswerprocess(self, query):
         
        messages=[
                    {"role": "system", "content": """You are a helpful assistant."""},
                    {"role": "user", "content": """Please provide the answer directly!!!
                                      Note:
                                      Please answer strictly according to the format(Answer: ??)

                                      Query: {query} 
                                      Answer: ??""".format(query=query)
                    },
                ]


        response=self._call_llm(messages=messages)

        try:
            judge_match = re.search(r'Answer:\s*(.*)', response)
            judge_answer = judge_match.group(1)
            return  judge_answer
            
        except Exception as e:
            return "I do not know"

    
    # def answerprocess(self, query, context):
         
    #     messages=[
    #                 {"role": "system", "content": """You are a helpful assistant."""},
    #                 {"role": "user", "content": """Please provide the answer based Information!!!
                                     
    #                                   Note:
    #                                   Please answer strictly according to the format(Answer: ??)
                     
    #                                   Information: {context} 
    #                                   Query: {query} 
                     
    #                                   Answer: ??""".format(query=query,context=context)
    #                 },
    #             ]
    #     response=self._call_llm(messages=messages)
    #     response = re.sub(r'<think\b[^>]*>.*?</think\s*>','',response,flags=re.DOTALL | re.IGNORECASE)
    #     try:
    #         judge_match = re.search(r'Answer:\s*(.*)', response)
    #         judge_answer = judge_match.group(1)
    #         return  judge_answer 
            
    #     except Exception as e:
    #         return "I do not know"

    # def AswerNeedprocess(self, query, context):
         
    #     messages=[
    #                 {"role": "system", "content": """You are a helpful assistant."""},
    #                 {"role": "user", "content": """Please ask the questions you need in order to answer this query.
                                      
    #                                   Note:
    #                                   Please answer strictly according to the format(Question: ??)
                                       
    #                                   Query: {query} 
    #                                   Information: {context} 
                                      
    #                                   Please ask the questions you need in order to answer this query.
    #                                   Question: ??""".format(query=query,context=context)
    #                 },
    #             ]
    #     response=self._call_llm(messages=messages)
    #     response = re.sub(r'<think\b[^>]*>.*?</think\s*>','',response,flags=re.DOTALL | re.IGNORECASE)

    #     try:
    #         judge_match = re.search(r'Question:\s*(.*)', response)
    #         return  judge_match
            
    #     except Exception as e:
    #         return "I do not know"
        
    
    
    # def Triple_extract(self,context,entities):

    #     messages=[
    #                 {"role": "system", "content": """You are an expert in information extraction. Extract all explicit (subject, predicate, object) triples about the given Entities from the following sentence
                        
    #                     Requirements:
    #                     Replace pronouns (e.g., "he", "she", "it", etc.) with the appropriate entity name. 
    #                    ["Albert Einstein", "made", "significant contributions to physics")]
    #                     """},
    #                 {"role": "user", "content": """
    #                     Text: Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.
    #                     Entities:  ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
    #                     """},
    #                 {"role": "assistant", "content": """
    #                     ["Radio City", "located in", "India"],
    #                     ["Radio City", "is", "private FM radio station"],
    #                     ["Radio City", "started on", "3 July 2001"],
    #                     ["Radio City", "plays songs in", "Hindi"],
    #                     ["Radio City", "plays songs in", "English"],
    #                     ["Radio City", "forayed into", "New Media"],
    #                     ["Radio City", "launched", "PlanetRadiocity.com"],
    #                     ["PlanetRadiocity.com", "launched in", "May 2008"],
    #                     ["PlanetRadiocity.com", "is", "music portal"],
    #                     ["PlanetRadiocity.com", "offers", "news"],
    #                     ["PlanetRadiocity.com", "offers", "videos"],
    #                     ["PlanetRadiocity.com", "offers", "songs"]
    #                     """},
    #                 {"role": "user", "content": """
    #                     Text: {context}
    #                     Entities: {entities}
    #                     """.format(entities=entities,context=context)
    #                 },
    #             ]

    #     response=self._call_llm(messages=messages,max_completion_tokens=1500)
        
    #     response = re.sub(r'<think\b[^>]*>.*?</think\s*>','',response,flags=re.DOTALL | re.IGNORECASE)
    #     print(response)

    #     try:
            
    #         judge_match = re.search(r'Output:\s*(.*)', response)
    #         judge_answer = judge_match.group(1)
    #         return  judge_answer 
    #     except Exception as e:
    #         return []
    #          Extract triples that directly involve the provided entities. 
    #                     Don't analyze too much, extract the triplets related to entities directly
    #                     Maintain the accuracy of meaning and context from the source text.
    #                     """},
    #                 {"role": "user", "content": """
    #                     Text: "Albert Einstein proposed the theory of relativity. He also made significant contributions to physics."
    #                     Entities: ["Albert Einstein", "the theory of relativity", "physics"]
    #                     """},
    #                 {"role": "assistant", "content": """
    #                     ["Albert Einstein", "proposed", "the theory of relativity"],
            
           
# if __name__=="__main__":  

#     agent=PoolAgent(model="Qwen3-14B",llm_base_url="http://localhost:12345/v1",llm_api_key="your-llm-api-key-here")

#     messages=[
#         {"role": "user", "content": "Hello!"}
#     ]
#     agent._call_llm(messages)

            



