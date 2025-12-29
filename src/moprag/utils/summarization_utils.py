import logging
import os
import re
import json
from typing import List,Dict
import nltk
import logging
from abc import ABC, abstractmethod
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .prompts import ner,triple_extraction,chunk_eneities_summary,story_summary


logger = logging.getLogger(__name__)

class BaseSummarizationModel(ABC):
    @abstractmethod
    def LLMaction(self, context):
        pass

class LLMSummarizationModel(BaseSummarizationModel):

    def __init__(self, model_name="model_name", llm_base_url="https://api.example.com/v1",llm_api_key="your-api-key-here",
                max_completion_tokens=1500,temperature=0,stop_sequence=None):
       
        self.model_name=model_name
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.max_completion_tokens=max_completion_tokens
        self.temperature=temperature
        self.stop_sequence=stop_sequence

        # Set up OpenAI client with custom base URL
        self.client = OpenAI(base_url=self.llm_base_url,api_key=self.llm_api_key,)
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def LLMaction(self,massage) -> str:
        try:
            # Call OpenAI API interface to generate summary
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=massage,
                max_completion_tokens=self.max_completion_tokens,
                stop=self.stop_sequence,
                temperature=self.temperature,
                frequency_penalty=0,
                presence_penalty=0,
                top_p=1,
            )
            response=response.choices[0].message.content
            response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)

            return response

        except Exception as e:
            print(f"An error occurred: {e}")
            return str(e)

    def LLMEntityExtract(self,context:str)-> list[str]:
        
        prompt=ner.ner_prompt_builder(context)
        response=self.LLMaction(prompt)
        try:
            entities_match = re.search(r'Entities:\s*(.*?)(?:\s*\n|$)', response)
            if entities_match:
                entities_str = entities_match.group(1)
                # 分割并去除空格
                entities_list = [e.strip() for e in entities_str.split(',') if e.strip()]
            else:
                entities_list = []

            return entities_list
        
        except Exception as e:
            print(f"Error parsing response: {e}")
            return []

    
    def LLMEntityExtract(self,context:str)-> list[str]:

        prompt=ner.ner_prompt_builder(context)
        response=self.LLMaction(prompt)
        try:
            entities_match = re.search(r'Entities:\s*(.*?)(?:\s*\n|$)', response)
            if entities_match:
                entities_str = entities_match.group(1)
                # 分割并去除空格
                entities_list = [e.strip() for e in entities_str.split(',') if e.strip()]
            else:
                entities_list = []

            return entities_list
        
        except Exception as e:
            print(f"Error parsing response: {e}")
            return []
        
    def LLMTripleextract(self,context,entities)-> list:
        messages=triple_extraction.teiple_prompt_builder(context,entities)
        response=self.LLMaction(messages) 
        response = re.sub(r'<think\b[^>]*>.*?</think\s*>','',response,flags=re.DOTALL | re.IGNORECASE)
        try:
            triples_str_list = re.findall(r'\[(.*?)\]', response)
            triples = []
            for t_str in triples_str_list:
               
                parts = [part.strip() for part in t_str.split(',')]
                if len(parts) == 3:  
                    triples.append(parts)
            # for i, triple in enumerate(triples, 1):
            #     print(f"{i}. {triple}")
            # print(triples)
            return triples
        except (ValueError, SyntaxError) as e:
            return []



    def LLMSummaryextract(self,context,entities)-> dict:
        messages=chunk_eneities_summary.entities_summary_prompt(context,entities)
        response=self.LLMaction(messages) 
        response = re.sub(r'<think\b[^>]*>.*?</think\s*>','',response,flags=re.DOTALL | re.IGNORECASE)
        
        try:
            result_dict = json.loads(response)
            # for i, triple in enumerate(triples, 1):
            #     print(f"{i}. {triple}")
            return result_dict
        except (ValueError, SyntaxError) as e:
            return {}

    
    def LLMStorySummaryextract(self,context)-> dict:
        messages=story_summary.story_summary_prompt(context)
        response=self.LLMaction(messages) 
        response = re.sub(r'<think\b[^>]*>.*?</think\s*>','',response,flags=re.DOTALL | re.IGNORECASE)
    
        return response



# if __name__=="__main__":
    
#     LLM=LLMSummarizationModel(model_name="Qwen3-14B", llm_base_url="http://localhost:12345/v1")

#     list=LLM.LLMEntityExtract("Jun is a Dog")
#     lista=LLM.LLMTripleextract("Jun is a Dog",list)
#     dict=LLM.LLMSummaryextract("Jun is a Dog",list)
#     print(dict.type)
        
  