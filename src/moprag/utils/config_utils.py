import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

@dataclass
class BaseConfig:
     
    llm_name: str = field(
        default="Qwen3-14B",
        metadata={"help": "Class name indicating which LLM model to use."}
    )

    llm_base_url: Optional[str] = field(
        default="http://localhost:12345/v1",
        metadata={"help": "Base URL for the vLLM model, if none, means using OPENAI service."}
    )
    
    llm_api_key: Optional[str] = field(
        default="your-llm-api-key-here",
        metadata={"help": "API key for the LLM model."}
    )

    max_completion_tokens: int =field(
        default=1500,
        metadata={"help":"path to your embedding model"}
    )
    temperature: int =field(
        default=0,
        metadata={"help":"path to your embedding model"}
    )
    stop_sequence: True | False = field(
        default=None,
        metadata={"help":"whether to chunk long context for LLM input"}
    )

    embedding_model_name: str = field(
        default="/data0/models/embeddingmodels/bge-base-en",
        metadata={"help":"path to your embedding model"}
    )

    llm_chunk: True | False = field(
        default=None,
        metadata={"help":"whether to chunk long context for LLM input"}
    )
    
    
    embedding_db_path: str = field(
        default="./data/moprag_text_embedding_db",
        metadata={"help":"path to your text embedding database"}
    )

    namepace: str = field(
        default="moprag_graph_node_db",
        metadata={"help":"path to your text embedding database"}
    )

    plot_namepace: str = field(
        default="moprag_graph_plot_node_db",
        metadata={"help":"path to your text embedding database"}
    )


    story_namepace: str = field(
        default="moprag_graph_story_node_db",
        metadata={"help":"path to your text embedding database"}
    )

    pic_path: str = field(
        default="./data/picture",
        metadata={"help":"path to your picture"}
    )

    memory_path: str = field(
        default="./data/memory",
        metadata={"help":"path to your memory"}
    )

    

    embedding_batch_size: int = field(
        default=16,
        metadata={"help":"batch size for embedding model inference"}
    )
    
    
    path_explorations_number: int = field(
        default=3,
        metadata={"help":"Number of path explorations"}
    )

    multi_round_number: int = field(
        default=2,
        metadata={"help":"Number of multi rounds"}
    )

    Triple_Tool: True | False = field(
        default=True,
        metadata={"help":"whether use hierarchical extraction"}
    )

    



    


    



    


