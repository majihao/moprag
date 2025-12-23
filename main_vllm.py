from openai import OpenAI
import json

from src.moprag.utils.config_utils import BaseConfig
from src.moprag.MopRAG import MopRAG


from pathlib import Path

def clear_directory(folder_path):
    folder = Path(folder_path)
    for file in folder.glob("*"):
        if file.is_file():
            file.unlink()  # 删除文件
        elif file.is_dir():
            
            pass  

clear_directory("/md0/home/majihao/MoPRAG/MopRAG0.2/data/moprag_text_embedding_db")

with open('/md0/home/majihao/MoPRAG/MopRAG/data/NarrativeQA/Nar_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

data=data[0]

document=data['corpus']


config=BaseConfig(
    llm_base_url="http://localhost:12344/v1",
    llm_api_key="your-llm-api-key-here",
    llm_name="Qwen3-14B",
    embedding_model_name="/data0/models/embeddingmodels/bge-base-en",
)



moprag=MopRAG(global_config=config)

doc=document

qal=[{}]

moprag.index(docs=doc)
answer=moprag.query("Mr. Kendrew argues that a man with a substantial inheritance needs a wife from high society to secure what?")

# moprag.query("Mr. Kendrew argues that a man with a substantial inheritance needs a wife from high society to secure what?")




# 指向你的 vLLM 服务地址
# client = OpenAI(
#     base_url=config.llm_base_url,
#     api_key=config.llm_api_key  # vLLM 不验证 key，随便填
# )



# response = client.chat.completions.create(
#     model=config.llm_name,  # 必须和启动时一致
#     messages=[
#         {"role": "user", "content": "Hello!"}
#     ],
#     max_tokens=512,
#     temperature=0.2
# )



# print(response.choices[0].message.content)