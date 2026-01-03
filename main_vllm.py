from openai import OpenAI
import json
import shutil
from src.moprag.utils.config_utils import BaseConfig
from src.moprag.MopRAG import MopRAG


from pathlib import Path

# def clear_directory(folder_path):
#     folder = Path(folder_path)
#     for file in folder.glob("*"):
#         if file.is_file():
#             file.unlink()  # 删除文件
#         elif file.is_dir():
            
#             pass 

def clear_directory(folder_path):
    folder = Path(folder_path)
    
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return
    
    for item in folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)  
        else:
            item.unlink()        
 

# clear_directory("/md0/home/majihao/MoPRAG/MopRAG0.2/data/moprag_text_embedding_db")

with open('/md0/home/majihao/MoPRAG/MopRAG/data/NarrativeQA/Nar_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

data=data[0]

document=data['corpus']


config=BaseConfig(
    llm_base_url="http://localhost:12345/v1",
    llm_api_key="your-llm-api-key-here",
    llm_name="Qwen3-14B",
    embedding_model_name="/data0/models/embeddingmodels/bge-base-en",
)



moprag=MopRAG(global_config=config)

doc=document

qal=[{}]

# moprag.index(docs=doc)
answer=moprag.query("Mr. Kendrew argues that a man with a substantial inheritance needs a wife from high society to secure what?")
print(answer)





