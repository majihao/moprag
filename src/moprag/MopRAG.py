from openai import OpenAI
import numpy as np
import heapq
import ast
import hashlib 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from types import SimpleNamespace

from .utils.embmodel_utils import embeddingmodel
from .embedding_store import GraphEmbeddingStore,Plot_GraphEmbeddingStore,Story_GraphEmbeddingStore
from .utils.chunk_utils import BChunkModel
from .utils.summarization_utils import LLMSummarizationModel
from .utils.memery_utils import Memory
from .utils.agent_utils import PoolAgent
from .rerank import VLLMWrapper, DSPyFilter

from tqdm import tqdm

def compute_mdhash_id(text: str, prefix: str = "") -> str:

    # Combine prefix and text, encode to bytes, and compute MD5 hash
    hashed = hashlib.md5((prefix + text).encode("utf-8")).hexdigest()
    return hashed


class MopRAG:
    def __init__(self, global_config):

        self.global_config = global_config
        self.path_explorations_number=global_config.path_explorations_number
        
        self.multi_round_number=global_config.multi_round_number
        
        self.Triple_Tool=global_config.Triple_Tool
        
        self.client = OpenAI(
            base_url=global_config.llm_base_url,
            api_key=global_config.llm_api_key
        )
        self.max_completion_tokens=global_config.max_completion_tokens
        self.temperature=global_config.temperature
        self.stop_sequence=global_config.stop_sequence

        self.question_type=global_config.question_type
        
        self.path_extect_slide_windows=global_config.path_extect_slide_windows_size

        self.embedding_model = embeddingmodel(
            model_path=global_config.embedding_model_name
        )

        
        narrtiverag = SimpleNamespace(
                llm_name=global_config.llm_name,  
                rerank_dspy_file_path=None,
                llm_model=VLLMWrapper(base_url=global_config.llm_base_url, model_name=global_config.llm_name, api_key=global_config.llm_api_key)
            )
        self.dspy_filter = DSPyFilter(narrtiverag)
          
        
        self.chunk_model=BChunkModel()

        self.summarizer=LLMSummarizationModel(
            model_name=global_config.llm_name,
            llm_base_url=global_config.llm_base_url,
            llm_api_key=global_config.llm_api_key,
            max_completion_tokens=global_config.max_completion_tokens,
            temperature=global_config.temperature,
            stop_sequence=global_config.stop_sequence
        )

        self.detail_embedding_store = GraphEmbeddingStore(
            embedding_model=self.embedding_model,
            db_filename=global_config.embedding_db_path,
            pic_filename=global_config.pic_path,
            batch_size=global_config.embedding_batch_size,
            namespace=global_config.namepace
        ) 

        self.plot_embedding_store = Plot_GraphEmbeddingStore(
            embedding_model=self.embedding_model,
            db_filename=global_config.embedding_db_path,
            pic_filename=global_config.pic_path,
            batch_size=global_config.embedding_batch_size,
            plot_namespace=global_config.plot_namepace
        ) 

        self.story_embedding_store = Story_GraphEmbeddingStore(
            embedding_model=self.embedding_model,
            db_filename=global_config.embedding_db_path,
            pic_filename=global_config.pic_path,
            batch_size=global_config.embedding_batch_size,
            story_namespace=global_config.story_namepace,
            SummarizerAgent=self.summarizer,
        ) 
        

        self.PoolAgent=PoolAgent(
            model=global_config.llm_name,
            client=self.client,
            max_completion_tokens=global_config.max_completion_tokens,
            stop_sequence=global_config.stop_sequence,
            temperature=global_config.temperature
        )
        
        self.memory=Memory(memory_path=global_config.memory_path,
                           agent= self.PoolAgent)



        
    def index(self, docs):

        chunks=self.chunk_model.chunk(context=docs, chunk_size= 11,overlap=2)  
        # print(len(chunks))
        entities=[]
        Triples=[]
        entities_dicts=[]
        # chunks=chunks[:20]
        for chunk in tqdm(chunks):
            entity=self.summarizer.LLMEntityExtract(
                context=chunk,)


            # Triple=self.summarizer.LLMTripleextract(chunk,entity)

            dict=self.summarizer.LLMSummaryextract(chunk,entity)
            entities.append(entity)
            # Triples.append(Triple)
            
            for key, value in dict.items():
                entities_dicts.append({key: value})
            # print(entities_dicts)
        # print(len(entities_dicts))
        
        self.detail_embedding_store.insert(chunks, entities)
        self.story_embedding_store.insert(chunks, entities)
        self.plot_embedding_store.insert(entities_dicts)

    def add_index(self):
        pass
     
     
    
    def query(self, query: str):

        if  self.PoolAgent.djudgprocess(query):
            print("can answer derestly!")
            return self.PoolAgent.danswerprocess(query)
        else: 
            print("need to reason path!")
             
            answer=self.Cognitive_control(query)
            
            return answer


    # def _load_Triple_graph_data(self,path,contents):

    #     print("load_Triple_graph_data")
        
    #     for idx,hashid in enumerate(path):
    #         content= self.content_data.loc[self.content_data['hash_id'] == hashid, 'content'].iloc[0] 
    #         entity= self.content_data.loc[self.content_data['hash_id'] == hashid, 'entity'].iloc[0] 
    #         Triple_data=self.PoolAgent.Triple_extract(content,entity)    
            
    #         print(Triple_data)


    def Muti_Round_Query(self, init_query,init_contents,round):
        
        query_list=[]
        init_query=init_query
        contents=init_contents

        round_Record=0
        while round_Record<round:

            print("-----------Round {}-----------".format(round_Record))
            query_str= "\n".join(query_list)
            self_prob_list=self.self_prob(init_query, query_str ,  contents)
            
            for subquery in self_prob_list:
                print(subquery)

                character_graph, character_contents, detail_graph, detail_contents,story_graph,story_contents=self._data_load() 
                
                
                character_paths=self.retrieve(query=subquery,graph=character_graph,contents= character_contents,top_k=5)
                detail_paths=self.retrieve(query=subquery,graph=detail_graph,contents= detail_contents,top_k=5)
                story_paths=self.retrieve(query=subquery,graph=story_graph,contents= story_contents,top_k=5)

                character_path_inf=self.path_ext(subquery,character_paths,character_graph,character_contents)
                detail_path_inf=self.path_ext(subquery,detail_paths,detail_graph,detail_contents)
                story_path_inf=self.path_ext(subquery,story_paths,story_graph,story_contents)

                print(character_path_inf,"\n",detail_path_inf)
                #mem_enc
                inf=self.mem_enc(subquery,detail_path_inf,character_path_inf,story_path_inf)
                print(inf)

                
            round_Record=round_Record+1
        
            #mem_dec
            inf=self.mem_dec(subquery)
            print(inf)
            answer=self.try_answer(subquery,inf,self.question_type)
            if answer is None:
                continue
            else:
                print("Final Answer: ",answer)
                break
        if answer is None:
            return "I don't know."
        else: 
            return answer


            
           
    

    def Cognitive_control(self, query:str):
        
        #path_ext
        character_graph, character_contents, detail_graph, detail_contents,story_graph,story_contents=self._data_load() 
        
        print("character_graph nodes:",len(character_graph))
        print("character_graph nodes:",len(character_contents))
        print("detail_graph nodes:",len(detail_graph))
        print("detail_graph nodes:",len(detail_contents))
        print("story_graph nodes:",len(story_graph))
        print("story_graph nodes:",len(story_contents))
        character_paths=self.retrieve(query=query,graph=character_graph,contents= character_contents,top_k=5)
        detail_paths=self.retrieve(query=query,graph=detail_graph,contents= detail_contents,top_k=5)
        story_paths=self.retrieve(query=query,graph=story_graph,contents= story_contents,top_k=5)
                  

        character_path_inf=self.path_ext(query,character_paths,character_graph,character_contents)
        detail_path_inf=self.path_ext(query,detail_paths,detail_graph,detail_contents)
        story_path_inf=self.path_ext(query,story_paths,story_graph,story_contents)
        
        print(character_path_inf,"\n",detail_path_inf,"\n",story_path_inf)
        
        #mem_enc
        inf=self.mem_enc(query,detail_path_inf,character_path_inf,story_path_inf)
        print(inf)
        #mem_dec

        inf=self.mem_dec(query)
        print(inf)

        answer=self.try_answer(query,inf,self.question_type)
        if answer is None:
            print("cannot answer derestly!")
            Manswer=self.Muti_Round_Query(query,inf,self.multi_round_number)
            return Manswer
        else:
            print("Final Answer: ",answer)
            return answer
        
        # return answer
        #try_answer
        # try_answer=self.PoolAgent.final_answer(init_query,inf)
    
    def self_prob(self,query, pre, cur):


        self_pro_list=self.PoolAgent.self_pro(query, pre, cur)

        return  self_pro_list
        

    def mem_dec(self,query):

        inf=self.memory.mem_pool_fuse(query)

        return inf

    def mem_enc(self,query, detail_path_inf,character_path_inf,story_path_inf):
        detail_path_inf = "" if detail_path_inf is None else detail_path_inf
        character_path_inf = "" if character_path_inf is None else character_path_inf
        story_path_inf = "" if story_path_inf is None else story_path_inf
        
        inf=self.PoolAgent.mem_enc(query ,detail_path_inf,character_path_inf,story_path_inf)

        self.memory.add_memory_pool(query, inf)

        return inf
    
    

    

    def path_ext(self,query,path,graph,contents):
        

        path_infs=""
        for node_path in path:
            inf=""
            for node in node_path:
                node_data=contents[contents['hash_id']==node]
                node_text=node_data['content'].values[0]
                inf+=node_text+"\n"
            path_inf=self.PoolAgent.path_ext(query,inf)
            path_infs=path_infs+path_inf+"\n"
            sum_path_inf=self.PoolAgent.path_ext(query,path_infs)
            self.memory.add_path_memory(query, path_inf, sum_path_inf)

        return sum_path_inf
           
    def _data_load(self): 
        
        character_graph=self.plot_embedding_store.graph
        character_contents=self.plot_embedding_store.contents
        
        detail_graph=self.detail_embedding_store.graph
        detail_contents=self.detail_embedding_store.contents

        story_graph=self.story_embedding_store.graph
        story_contents=self.story_embedding_store.contents

        return character_graph, character_contents, detail_graph, detail_contents,story_graph,story_contents



    def retrieve(self, query: str, graph,contents,top_k: int):

        query_entity=self.summarizer.LLMEntityExtract(context=query)
       
        pruned_graph,pruned_contents=self.Prune_Weaver(query=query,query_eneities=query_entity,graph=graph,contents=contents)

        
        paths=self.Path_Cognizer(query=query,graph=pruned_graph,contents=pruned_contents,toprank=top_k)

        print("---------paths---------")
        print(paths)
        
        paths=self.merge_path_edges(paths,self.path_extect_slide_windows)

        print(paths)

        return paths
    
    def Prune_Weaver(self,query:str,query_eneities:list[str],graph,contents):
        
        graph,contents=self._prune_by_entity_alignment_extended(query,query_eneities,graph,contents)

        graph,contents=self.judge_graph_contents(graph,contents)

        graph,contents=self._prune_by_similar_and_center(query,graph,contents,normalize_scores=True, alpha=0.6, beta=0.4,pruning_rate=0.3)
        
        graph,contents=self.judge_graph_contents(graph,contents)
        
        return graph, contents
        # pos = nx.spring_layout(self.graph_data, seed=42)  # 固定 seed 保证可复现

        # nx.draw_networkx_nodes(self.graph_data, pos, node_size=700, node_color="lightblue")
        # nx.draw_networkx_labels(self.graph_data, pos)

        # share_edges = [(u, v) for u, v, d in self.graph_data.edges(data=True) if d.get('relation') == "share_entities"]
        # next_edges = [(u, v) for u, v, d in self.graph_data.edges(data=True) if d.get('relation') == 'next']

        # nx.draw_networkx_edges(self.graph_data, pos, edgelist=share_edges, edge_color='blue', style='solid', alpha=0.7)  
        # nx.draw_networkx_edges(self.graph_data, pos, edgelist=next_edges, edge_color='green', style='dashed', alpha=0.8)
        # plt.savefig("./data/picture/my_plot2.png", dpi=300, bbox_inches='tight')
        # plt.show()
    

    
    def Path_Cognizer(self,query:str,graph,contents,toprank:int):
       
        
        graph_emb=contents['embedding'].tolist()
        query_emb = self.embedding_model.batch_encode(query)
        query_emb = np.array(query_emb).reshape(1, -1)
        embs_arr = np.array(graph_emb)
        sim_scores = cosine_similarity(query_emb, embs_arr).flatten()
        hashid=contents['hash_id'].tolist()
        node_query_sim = dict(zip(hashid, sim_scores))
        
        top2 = sorted(node_query_sim.items(), key=lambda x: x[1], reverse=True)[0:toprank+3]

        top2_nodes = [node for node, score in top2]
        #rerank
        # top2 = sorted(node_query_sim.items(), key=lambda x: x[1], reverse=True)[0:toprank]
        # print("top2",top2)

        # top2_nodes = [node for node, score in top2]
        Rrerank_nodes=self.rerank(query,top2_nodes,contents,toprank)

        top2_sorted_by_appearance = [node for node in hashid if node in set(Rrerank_nodes)]
        
        path=[]
        for i in range(len(top2_sorted_by_appearance)-1):
            result = self._semantic_path_selection(query=query,graph=graph,contents=contents,source=top2_sorted_by_appearance[i],target=top2_sorted_by_appearance[i+1])
            
            # print(result[0][0])
            path.append(result[0][0])

        return path

    
    def _semantic_path_selection(self,query,graph,contents,source,target,alpha=0.5,beta=0.5,max_path_len=4,top_k=1,decay_lambda=0.5):

        
        temp_graph_data=graph

        if source == target:
            return [([source], 1.0)]
        graph_emb=contents['embedding'].tolist()
        
        query_emb = self.embedding_model.batch_encode(query)
        query_emb = np.array(query_emb).reshape(1, -1)
        embs_arr = np.array(graph_emb)
        sim_scores = cosine_similarity(query_emb, embs_arr).flatten()
        hashid=contents['hash_id'].tolist()
        node_query_sim = dict(zip(hashid, sim_scores))
  
        best_results = []
        priority_queue = [(-1.0, 1, source, [source])]
       
        while priority_queue:
            
            neg_score, length, current, path = heapq.heappop(priority_queue)

            # print(f"当前路径: {path}, 候选邻居: {list(temp_graph_data.neighbors(current))}")
    
            if length > max_path_len:
                break
            if len(path) != len(set(path)):      
                continue

            if current == target and len(path) >= 2:
                
                node_sims = [node_query_sim[n] for n in path]
                avg_node_sim = np.mean(node_sims)
              
                edge_flows = [
                    self.edge_flow(path[i], path[i + 1],contents)
                    for i in range(len(path) - 1)
                ]
                avg_edge_flow = np.mean(edge_flows) if edge_flows else 0.0

                base_score = alpha * avg_node_sim + beta * avg_edge_flow
                
                # 4. 应用路径长度衰减
                num_edges = len(path) - 1  # 路径边数
                decay_factor = np.exp(-decay_lambda * num_edges)
                final_score = base_score * decay_factor
                
                best_results.append((final_score, path))
                if len(best_results) > top_k * 5:
                    best_results.sort(key=lambda x: x[0], reverse=True)
                    best_results = best_results[:top_k * 3]
                continue
            
            idx=0
            for neighbor in temp_graph_data.neighbors(current):
                if neighbor in path:
                    idx+=1
                    if idx>20:
                        break
                    # 防环
                    continue  
                new_path = path + [neighbor]
                new_len = length + 1
                heapq.heappush(priority_queue, (-neg_score, new_len, neighbor, new_path))
            
            # temp_graph_data.remove_node(current)


        seen = set()
        final = []
        best_results.sort(key=lambda x: x[0], reverse=True)    
        
        for score, path in best_results:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                final.append((path, score))
                if len(final) >= top_k:
                    break
        
        if not final:
            return [([source, target], 0.0)]

        return final
            
    def edge_flow(self,nodeA, nodeB,content_data):

        rowA = content_data[content_data['hash_id'] == nodeA]
        rowB = content_data[content_data['hash_id'] == nodeB]

        # print(rowA["embedding"].tolist(),rowB["embedding"].tolist()) 
        sim = cosine_similarity(rowA["embedding"].tolist(), rowB["embedding"].tolist())
        return sim 


    def _prune_by_entity_alignment_extended(self,query,query_entity,graph,contents):

        
        target_nodes=[]
        
        similar_nodes=self.similar_top_K(query_emb=self.embedding_model.batch_encode(query),
                                        emb_list=contents['embedding'].tolist(),
                                        hashid_list=contents['hash_id'].tolist(),
                                        top_k=10
                                         )
        target_nodes.extend(similar_nodes)


        eneity_target_nodes = []
        for node, attr in graph.nodes(data=True):
            if set(query_entity) & set(attr.get("entity", "")):
                eneity_target_nodes.append(node)
        target_nodes.extend(eneity_target_nodes)
        
    
        next_neighbors=[]
        for eneity_target_node in eneity_target_nodes:
            next_neighbor = [
                nbr for nbr in graph.neighbors(eneity_target_node)
                if graph[eneity_target_node][nbr].get('relation') == 'next'
            ]
            next_neighbors.extend(next_neighbor)
        target_nodes.extend(next_neighbors)
        
    
        target_nodes = list(set(target_nodes))

        graph = graph.subgraph(target_nodes).copy()
        contents = contents[contents['hash_id'].isin(target_nodes)]

        
        return graph, contents



    def _prune_by_similar_and_center(self,query,graph, contents,normalize_scores=True, alpha=0.6, beta=0.4,pruning_rate=0.3):
        
        if len(graph) == 0:
            print("none graph data")
       
        graph_emb=contents['embedding'].tolist()
        
        query_emb = self.embedding_model.batch_encode(query)
        query_emb = np.array(query_emb).reshape(1, -1)
        embs_arr = np.array(graph_emb)
        sim_scores = cosine_similarity(query_emb, embs_arr).flatten()  # shape: (n,)
        
        valid_nodes = []
        pagerank_dict = nx.pagerank(graph, alpha=0.85)  # 阻尼因子 0.85 是标准值
        pr_scores = np.array([pagerank_dict[node] for node in graph]) 

        if normalize_scores: 
            scaler=MinMaxScaler()
            sim_norm = scaler.fit_transform(sim_scores.reshape(-1, 1)).flatten()
            pr_norm = scaler.fit_transform(pr_scores.reshape(-1, 1)).flatten()
        else:
            sim_norm = sim_scores
            pr_norm = pr_scores

        
        scores = alpha * sim_norm + beta * pr_norm
        
        node_score_map = dict(zip(graph, scores))

        sorted_nodes = sorted(node_score_map.items(), key=lambda x: x[1], reverse=False)
        
        selected_nodes = [node for node, _ in sorted_nodes[int(len(sorted_nodes)*pruning_rate):]]
        
        graph = graph.subgraph(selected_nodes).copy()
        contents= contents[contents['hash_id'].isin(selected_nodes)]

        return graph, contents

        
    def similar_top_K(self, query_emb,emb_list,hashid_list,top_k):

        query_emb = np.array(query_emb).reshape(1, -1)  # shape: (1, d)

        emb_array = np.array(emb_list)                  # shape: (n, d)

        similarities = cosine_similarity(query_emb, emb_array).flatten()  # shape: (n,)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        top_hash_ids = [hashid_list[i] for i in top_indices]

        return top_hash_ids

    def Path_concatenation(self, path_segments):
        if not path_segments:
            return []
        merged = path_segments[0][:] 
        for segment in path_segments[1:]:
            if merged and segment and merged[-1] == segment[0]:
                
                merged.extend(segment[1:])
            else:             
                raise ValueError(f"Path segments not contiguous: {merged[-1]} != {segment[0]}")
        return merged
    
    def try_answer(self,query,context,question_type):
        
        if question_type=="QA":
            answer=self.PoolAgent.try_answer(query,context)
            
            if "No"  in answer:
                return None
            return answer
        
        if question_type=="MC":
            answer=self.PoolAgent.try_answer_MC(query,context)
            
            if "No"  in answer:
                return None
            return answer

    
    def rerank(self,query,hashids,contents,top_K)->list:
        print("---------rerank---------")
        results=[]
        content_lists=contents[contents['hash_id'].isin(hashids)]["content"].tolist()
        entity_lists=contents[contents['hash_id'].isin(hashids)]["entity"].tolist()
        for i in range(len(content_lists)):
            Triple=self.summarizer.LLMTripleextract(content_lists[i],entity_lists[i])
            Triple_indies=list(range(len(Triple)))

            filtered_indices, filtered_facts, meta=self.dspy_filter.rerank(query,Triple,Triple_indies,len_after_rerank=5)
            results.append(len(filtered_facts))
        
        paired = list(zip(hashids, results))
        topk_stable = sorted(paired, key=lambda x: x[1], reverse=True)[:top_K]

        topk_hashids = [h for h, s in topk_stable]

        return topk_hashids
    


    def merge_connected_paths_safe(self, paths, path_extect_slide_windows)->list:
        next_map = {}
        prev_map = {}
        all_nodes = set()


        for p in paths:
            for i in range(len(p) - 1):
                # 如果已有映射，说明有分叉或冲突（可选报错）
                if p[i] in next_map and next_map[p[i]] != p[i+1]:
                    print(f"Warning: {p[i]} has multiple successors!")
                next_map[p[i]] = p[i+1]
                prev_map[p[i+1]] = p[i]
                all_nodes.add(p[i])
            all_nodes.add(p[-1])

        # 找起点（无前驱）
        starts = [n for n in all_nodes if n not in prev_map]

        merged = []
        for start in starts:
            path = []
            current = start
            visited = set()
            while current is not None and current not in visited:
                visited.add(current)
                path.append(current)
                current = next_map.get(current)
            merged.append(path)
        
        merged = merged[0]
        merged = [merged[i:i+path_extect_slide_windows] for i in range(0, len(merged), path_extect_slide_windows)]

        return merged

    # def query(self, docs):


    def merge_path_edges(self, paths, path_extect_slide_windows)->list:
        if not paths:
            return []
        else:
            lst=[]
            for item in paths:
                for i in item:
                    lst.append(i)
            path = list(dict.fromkeys(lst))
            merged = [path[i:i+path_extect_slide_windows] for i in range(0, len(path), path_extect_slide_windows)]

            return merged

    def judge_graph_contents(self,graph,contents):
        graph_nodes = set(graph.nodes)
        content_hashids = set(contents['hash_id'])
        if graph_nodes == content_hashids:

            return graph,contents
        else:
            missing_in_graph = content_hashids - graph_nodes      # 在 df 里但不在图中
            missing_in_contents = graph_nodes - content_hashids   # 在图中但不在 df 里

            if missing_in_graph:
                valid_hashids = set(graph.nodes)
                contents_cleaned = contents[contents['hash_id'].isin(valid_hashids)].reset_index(drop=True)
                contents=contents_cleaned
            if missing_in_contents:
                valid_hashids = set(contents['hash_id'])
                graph = [node for node in graph.nodes if node not in valid_hashids]
            
            return graph,contents

        # else:# 构建 next 映射 和 所有节点集合
        #     next_map = {}
        #     all_nodes = set()
        #     second_nodes = set()  # 所有作为后继（第二个元素）的节点

        #     for a, b in paths:
        #         next_map[a] = b
        #         all_nodes.add(a)
        #         all_nodes.add(b)
        #         second_nodes.add(b)

        #     # 起点：在 all_nodes 中，但不在 second_nodes 中
        #     start_candidates = all_nodes - second_nodes
        #     if len(start_candidates) != 1:
        #         raise ValueError(f"Expected exactly one start node, got: {start_candidates}")
            
        #     start = start_candidates.pop()
            
        #     # 从起点开始重建完整路径
        #     path = [start]
        #     current = start
        #     while current in next_map:
        #         current = next_map[current]
        #         path.append(current)
            
        #     merged = [path[i:i+path_extect_slide_windows] for i in range(0, len(path), path_extect_slide_windows)]

            
        #     return merged




    
    
                 
    