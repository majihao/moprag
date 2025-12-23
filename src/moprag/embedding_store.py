import numpy as np
import os
import pandas as pd 
from typing import List, Dict, Optional, Any
import logging
import hashlib 
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict



# from .utils.embmodel_utils import embeddingmodel

logger = logging.getLogger(__name__)

def compute_mdhash_id(text: str, prefix: str = "") -> str:

    # Combine prefix and text, encode to bytes, and compute MD5 hash
    hashed = hashlib.md5((prefix + text).encode("utf-8")).hexdigest()
    return hashed


class GraphEmbeddingStore:

    def __init__(self, embedding_model: str,db_filename: str, pic_filename: str,batch_size: int, namespace: str):
        
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)
        
        if not os.path.exists(pic_filename):
            logger.info(f"Creating working directory: {pic_filename}")
            os.makedirs(pic_filename, exist_ok=True)

        self.filename_graph = os.path.join(
            db_filename, f"vdb_{self.namespace}.pkl"
        )

        self.filename_content = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        
        self._load_data()
      
    def _load_data(self):
        
        if os.path.exists(self.filename_graph):

            with open(self.filename_graph, "rb") as f:
                self.graph = pickle.load(f)
            self.contents = pd.read_parquet(self.filename_content)
        else:
            self.graph = nx.Graph()
            self.contents = pd.DataFrame(columns=["hash_id","content","embedding","entity"])
    
    def insert(self,texts: List[str], entites: List[str]):

        texts_embeddings = self.embedding_model.batch_encode(texts)
        
        
        for idx, text in enumerate(texts):
            
            hashid=compute_mdhash_id(text, prefix="")

            if not self.graph.has_node(hashid):

                self.graph.add_node(hashid, 
                                    content=text, 
                                    embedding=texts_embeddings[idx].tolist(), 
                                    entity=entites[idx],
                                    type="text"
                                    )
                
                self.contents = pd.concat([self.contents, pd.DataFrame([{
                    "hash_id": hashid, 
                    "content": text,
                    "embedding": texts_embeddings[idx].tolist(),
                    "entity": entites[idx]
                    }])], ignore_index=True)
            

            else:
                continue
            
        self._connect_nodes_by_entities()
        
        hash_ids = self.contents['hash_id'].tolist()
        for i in range(len(hash_ids) - 1):
            self.graph.add_edge(
                hash_ids[i], 
                hash_ids[i + 1], 
                relation="next"
            )

        self._save_data()

        # for u, v, data in self.graph.edges(data=True):
        #     print(f"Edge: {u} -> {v}, Attributes: {data}")


        pos = nx.spring_layout(self.graph)  # 固定 seed 保证可复现

      
        nx.draw_networkx_nodes(self.graph, pos, node_size=50, node_color="lightblue")
        nx.draw_networkx_labels(self.graph, pos)

      
        share_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('relation') == "share_entities"]
        next_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('relation') == 'next']

       
        nx.draw_networkx_edges(self.graph, pos, edgelist=share_edges, edge_color='blue', style='solid', alpha=0.7)

       
        nx.draw_networkx_edges(self.graph, pos, edgelist=next_edges, edge_color='green', style='dashed', alpha=0.8)

        plt.savefig("./data/picture/my_plot.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    
    def add_insert(self,texts: List[str], entites: List[str]):
        
        texts_embeddings = self.embedding_model.batch_encode(texts)
        
        for idx, text in enumerate(texts):

            hashid=compute_mdhash_id(text, prefix="")
            self.graph.add_node(hashid, 
                                content=text, 
                                embedding=texts_embeddings[idx].tolist(),  
                                entity=entites[idx],
                                type="text"
                                )
            
            self.contents = pd.concat([self.contents, pd.DataFrame([{
                "hash_id": hashid, 
                "content": text,
                "embedding": texts_embeddings[idx].tolist(),
                "entity": entites[idx]
                }])], ignore_index=True)
            
        
        self._connect_nodes_by_entities()
        
        self._save_data()

    
    
    def _connect_nodes_by_entities(self):
        
        added_edges = set()
        
        for node, data in self.graph.nodes(data=True):
            
            entity = data.get("entity", "")
            if not entity:
                continue

            for other_node, other_data in self.graph.nodes(data=True):
                
                if node == other_node:
                    continue
                other_entity = other_data.get("entity", "")

                if set(entity) & set(other_entity):
                
                    common = list(set(entity) & set(other_entity))

                    edge_key = tuple(sorted([node, other_node]))
                    if edge_key not in added_edges:
                        self.graph.add_edge(
                            node, 
                            other_node, 
                            shared_entities=common,      # 更清晰的属性名
                            relation="share_entities"
                        )
                        added_edges.add(edge_key)
                
            
    def _save_data(self):

        self.contents.to_parquet(self.filename_content, index=False)

        with open(self.filename_graph, "wb") as f:
            pickle.dump(self.graph, f)

    def get_number_of_node(self,hashid):

        positions = self.contents.index[self.contents['hash_id'] == hashid].tolist()
        
        return positions

        


class Plot_GraphEmbeddingStore:

    def __init__(self, embedding_model: str,db_filename: str, pic_filename: str,batch_size: int, plot_namespace: str):
        
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = plot_namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)
        
        if not os.path.exists(pic_filename):
            logger.info(f"Creating working directory: {pic_filename}")
            os.makedirs(pic_filename, exist_ok=True)

        self.filename_graph = os.path.join(
            db_filename, f"vdb_{self.namespace}.pkl"
        )

        self.filename_content = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        
        self._load_data()
    
    def _load_data(self):
        
        if os.path.exists(self.filename_graph):

            with open(self.filename_graph, "rb") as f:
                self.graph = pickle.load(f)
            self.contents = pd.read_parquet(self.filename_content)
        else:
            self.graph = nx.Graph()
            self.contents = pd.DataFrame(columns=["hash_id","content","embedding","entity"])
    

    def insert(self,dicts:dict):
        
        for item in dicts:
            if item:  
                key = next(iter(item))
                value = item[key]
                hashid=compute_mdhash_id(value, prefix="")
                if not self.graph.has_node(hashid):
                    texts_embedding = self.embedding_model.batch_encode(value)
                    # print("plot",texts_embedding)
                
                    self.graph.add_node(hashid, 
                                        content=value, 
                                        embedding=texts_embedding[0].tolist(),  
                                        entity= key,
                                        type="text"
                                        )

                    last_hashid=self.get_last_hashid_for_entity(key)

                    self.contents = pd.concat([self.contents, pd.DataFrame([{
                        "hash_id": hashid, 
                        "content": value,
                        "embedding": texts_embedding[0].tolist(),
                        "entity":  key
                        }])], ignore_index=True)
                    

                    if last_hashid and last_hashid != hashid:
                        self.graph.add_edge(
                            last_hashid, 
                            hashid, 
                            relation="entities_plot",
                            entities=key     
                        )
                else:
                    continue
                    
        self._save_data()


        # print(len(self.graph))
        pos = nx.spring_layout(self.graph)  #   
        nx.draw_networkx_nodes(self.graph, pos, node_size=50, node_color="lightblue")
        # nx.draw_networkx_labels(self.graph, pos)
        share_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('relation') == "entities_plot"]
       
        nx.draw_networkx_edges(self.graph, pos, edgelist=share_edges, edge_color='red', style='solid', alpha=0.7)

        plt.savefig("./data/picture/my_plot2.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    
    def get_last_hashid_for_entity(self, entity, entity_col="entity", hashid_col="hash_id"):
        
        subset = self.contents[self.contents[entity_col] == entity]
        if subset.empty:
            return None
        last_row = subset.iloc[-1]  
        return last_row[hashid_col]
    
    # def _connect_nodes_by_entities(self):
        
    #     pass
    def _save_data(self):

        self.contents.to_parquet(self.filename_content, index=False)

        with open(self.filename_graph, "wb") as f:
            pickle.dump(self.graph, f)
            
    
    def Sliding_window_summary(self):

        pass
    




class Story_GraphEmbeddingStore:

    def __init__(self, embedding_model: str,db_filename: str, pic_filename: str,batch_size: int, story_namespace: str,SummarizerAgent):
        
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = story_namespace
        self.summarizer = SummarizerAgent

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)
        
        if not os.path.exists(pic_filename):
            logger.info(f"Creating working directory: {pic_filename}")
            os.makedirs(pic_filename, exist_ok=True)

        self.filename_graph = os.path.join(
            db_filename, f"vdb_{self.namespace}.pkl"
        )

        self.filename_content = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        
        self._load_data()
    
    def _load_data(self):
        
        if os.path.exists(self.filename_graph):

            with open(self.filename_graph, "rb") as f:
                self.graph = pickle.load(f)
            self.contents = pd.read_parquet(self.filename_content)
        else:
            self.graph = nx.Graph()
            self.contents = pd.DataFrame(columns=["hash_id","content","embedding","entity"])
    

    def insert(self,texts: List[str], entites: List[str], merge_size=8):
        print(len(self.graph),len(self.contents))
        texts_embeddings = self.embedding_model.batch_encode(texts)
        print(len(texts_embeddings))
       
        for idx, text in enumerate(texts):
            
            hashid=compute_mdhash_id(text, prefix="")

            if not self.graph.has_node(hashid):

                self.graph.add_node(hashid, 
                                    content=text, 
                                    embedding=texts_embeddings[idx].tolist(), 
                                    entity=entites[idx],
                                    type="text"
                                    )
                
                self.contents = pd.concat([self.contents, pd.DataFrame([{
                    "hash_id": hashid, 
                    "content": text,
                    "embedding": texts_embeddings[idx].tolist(),
                    "entity": entites[idx]
                    }])], ignore_index=True)
            

            else:
                continue
        
        print("story",len(self.contents))   
        print("story",len(self.graph))
            
        hash_ids = self.contents['hash_id'].tolist()
        for i in range(len(hash_ids) - 1):
            self.graph.add_edge(
                hash_ids[i], 
                hash_ids[i + 1], 
                relation="next"
            )
        
        for i in range(0, len(self.contents), merge_size):
            merge_text=""
            node_ids=[]
            entites=[]
            for idx, row in self.contents.iloc[i:i+merge_size].iterrows():
                
                merge_text=self.merge_two_chunks(merge_text,row["content"])
                node_ids.append(row["hash_id"])
                entites.extend(row["entity"])

            summery=self.summarizer.LLMStorySummaryextract(merge_text)
            hashid=compute_mdhash_id(summery, prefix="")
            
            texts_embedding = self.embedding_model.batch_encode(summery)
            
            self.graph.add_node(hashid, 
                                    content=summery, 
                                    embedding=texts_embedding[0].tolist(), 
                                    entity=entites,
                                    type="text"
                                    )
            
            self.contents = pd.concat([self.contents, pd.DataFrame([{
                "hash_id": hashid, 
                "content": summery,
                "embedding": texts_embedding[0].tolist(),
                "entity": entites
                }])], ignore_index=True)
        
            for node_id in node_ids:
                self.graph.add_edge(
                    hashid, 
                    node_id, 
                    relation="summery"
                )
        
        print("story",len(self.contents))   
        print("story",len(self.graph))
        self._save_data()

        # for u, v, data in self.graph.edges(data=True):
        #     print(f"Edge: {u} -> {v}, Attributes: {data}")


        pos = nx.spring_layout(self.graph)  # 固定 seed 保证可复现

      
        nx.draw_networkx_nodes(self.graph, pos, node_size=50, node_color="lightblue")
        nx.draw_networkx_labels(self.graph, pos)

      
        share_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('relation') == "summery"]
        next_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('relation') == 'next']

       
        nx.draw_networkx_edges(self.graph, pos, edgelist=share_edges, edge_color='blue', style='solid', alpha=0.7)

       
        nx.draw_networkx_edges(self.graph, pos, edgelist=next_edges, edge_color='green', style='dashed', alpha=0.8)

        plt.savefig("./data/picture/my_plot3.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    
     
    def merge_two_chunks(self, chunk1: str, chunk2: str) -> str:

        if not chunk1:
            return chunk2
        if not chunk2:
            return chunk1

        max_overlap = 0
        min_len = min(len(chunk1), len(chunk2))
        
        for i in range(min_len, 0, -1):
            if chunk1[-i:] == chunk2[:i]:
                max_overlap = i
                break  
          
        return chunk1 + chunk2[max_overlap:]


    def _save_data(self):

        self.contents.to_parquet(self.filename_content, index=False)

        with open(self.filename_graph, "wb") as f:
            pickle.dump(self.graph, f)
            
    
    def Sliding_window_summary(self):

        pass


# if __name__ == "__main__":
    
#     text="Hello, this is a sample text for hashing."

#     a=compute_mdhash_id(text, prefix="")

#     print(a)

#     contents = pd.DataFrame(columns=["hash_id","content","embedding","summary","entity"])
#     print(contents)