import os
import logging
from .agent_utils import PoolAgent

logger = logging.getLogger(__name__)

class Memory:
    def __init__(self, memory_path,capacity=10):
        
        if not os.path.exists(memory_path):
            logger.info(f"Creating memory directory: {memory_path}")
            os.makedirs(memory_path, exist_ok=True)

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
        
       


    def add_medium_memory(self, query , Information):     

        self.medium_memory.append({"query":query,"Information":Information})
        print(self.medium_memory)
        self.short_memory.clear()
           

# # 示例使用
# memory = Memory(capacity=5)
# memory.add_memory("我昨天去了公园")
# memory.add_memory("今天天气很好")

# print(memory.retrieve_memory("天气"))