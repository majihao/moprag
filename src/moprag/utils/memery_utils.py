import os
import re
import logging
from .agent_utils import PoolAgent
from .prompts.mem_dec import mem_dec_prompt

logger = logging.getLogger(__name__)

class Memory:
    def __init__(self, memory_path,agent,capacity=10):
        
        if not os.path.exists(memory_path):
            logger.info(f"Creating memory directory: {memory_path}")
            os.makedirs(memory_path, exist_ok=True)

        self.agent=agent

        self._memory_load()

    def _memory_load(self):
        self.path_memory=[]
        self.memory_pool=[]
    
    
    def add_long_memory(self, entry):
     
        pass
    
    def add_path_memory(self, query, path_inf, sum_path_inf):
        
        self.path_memory.append({"query":query,"path_inf":path_inf,"sum_path_inf":sum_path_inf})
        
        pass

    def add_memory_pool(self, query, path_inf):
        
        self.path_memory.append({"query":query,"inf":path_inf,})
        self.path_memory.clear()
        pass
        

    def mem_pool_fuse(self,init_query):
        
        prompt=""
        for item in self.path_memory:
            prompt=prompt+"[Q] "+item["query"] +"\n" + "[A] "+item["inf"]+"\n"
        
        messages=mem_dec_prompt(init_query,prompt)

        response = self.agent._call_llm(messages=messages)
        response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        
        return response


        

           





# # 示例使用
# memory = Memory(capacity=5)
# memory.add_memory("我昨天去了公园")
# memory.add_memory("今天天气很好")

# print(memory.retrieve_memory("天气"))