import logging
import os
import re
import json
from typing import List,Dict
import nltk
from nltk.tokenize import sent_tokenize
import logging
from abc import ABC, abstractmethod
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential



nltk.download('punkt')
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


import math

def compute_window_size(N):
    if N <= 0:
        raise ValueError("N mast > 0")  
    log2_N = math.log2(N)
    w = int(math.floor(log2_N * 2))  
    w = max(10, w)                   
    w = min(20, w)                      
    return w

class BaseChunkModel(ABC):
    @abstractmethod
    def chunk(self, context,**kwargs):
        pass

class BChunkModel(BaseChunkModel):

    def __init__(self):
        """
        Initialize class with support for custom model and API base URL.
        
        :param model: Model name
        :param llm_base_url: Base URL for LLM API
        """

    def chunk(self, context: str, chunk_size: int = 15, overlap: int = 2) -> list[str]:
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(context)
        chunk_size=compute_window_size(len(sentences))
        if chunk_size <= overlap:
            raise ValueError("chunk_size mast > overlap")  
        chunks = []
        step = chunk_size - overlap       
        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i : i + chunk_size]
            if not chunk_sentences:
                continue          
            chunk_text = " ".join(chunk_sentences)  
            chunks.append(chunk_text)
            if i + chunk_size >= len(sentences):
                break
        chunks = [s.replace('\n', ' ') for s in chunks]
        
        return chunks



        