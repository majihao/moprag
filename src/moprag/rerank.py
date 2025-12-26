import json
import difflib
from pydantic import BaseModel, Field, TypeAdapter
from typing import List, Tuple, Optional, Dict, Any
from copy import deepcopy
import re
import ast

# from .utils.prompts.filter_default_prompt import best_dspy_prompt  # 确保路径正确
best_dspy_prompt = {
    "prog": {
        "system": (
            "You are an expert fact filtering assistant. "
            "Given a question and a list of candidate facts (each as [subject, predicate, object]), "
            "select ONLY the facts that are directly relevant and correct for answering the question. "
            "Return the filtered facts in the exact format specified. "
            "Do not add, modify, or rephrase any fact. "
            "If no facts are relevant, return an empty list."
            "Note: Don't overthink, just filter based on direct relevance to the question."
        ),
        "demos": [
            {
                "question": "Who founded Microsoft?",
                "fact_before_filter": '{"fact": [["Bill Gates", "founded", "Microsoft"], ["Steve Jobs", "founded", "Apple"], ["Satya Nadella", "is CEO of", "Microsoft"]]}',
                "fact_after_filter": '{"fact": [["Bill Gates", "founded", "Microsoft"]]}'
            },
            {
                "question": "What company does Tim Cook lead?",
                "fact_before_filter": '{"fact": [["Tim Cook", "is CEO of", "Apple"], ["Elon Musk", "is CEO of", "Tesla"], ["Sundar Pichai", "is CEO of", "Google"]]}',
                "fact_after_filter": '{"fact": [["Tim Cook", "is CEO of", "Apple"]]}'
            },
            {
                "question": "Where is the Eiffel Tower located?",
                "fact_before_filter": '{"fact": [["Eiffel Tower", "is located in", "Paris"], ["Statue of Liberty", "is located in", "New York"], ["Great Wall", "is located in", "China"]]}',
                "fact_after_filter": '{"fact": [["Eiffel Tower", "is located in", "Paris"]]}'
            },
            {
                "question": "Who invented the telephone?",
                "fact_before_filter": '{"fact": [["Alexander Graham Bell", "invented", "telephone"], ["Thomas Edison", "invented", "light bulb"], ["Nikola Tesla", "worked on", "alternating current"]]}',
                "fact_after_filter": '{"fact": [["Alexander Graham Bell", "invented", "telephone"]]}'
            }
        ]
    }
}





class Fact(BaseModel):
    """Pydantic 模型：定义 LLM 应输出的事实结构"""
    fact: List[List[str]] = Field(
        description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]"
    )


def safe_literal_eval(s: str) -> Any:
    """安全地将字符串转为 Python 对象，避免 eval 风险"""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def match_triple_to_candidates(
    generated_triple: List[str],
    candidate_items: List[Tuple[str, str, str]],
    threshold: float = 0.6
) -> Optional[int]:
    """
    将 LLM 生成的三元组模糊匹配到原始候选列表中的索引。
    
    分别对 subject, predicate, object 做 fuzzy match，综合判断。
    """
    if len(generated_triple) != 3:
        return None

    best_idx, best_score = -1, 0.0

    for idx, (s, p, o) in enumerate(candidate_items):
        # 计算每个字段的相似度
        score_s = difflib.SequenceMatcher(None, generated_triple[0], s).ratio()
        score_p = difflib.SequenceMatcher(None, generated_triple[1], p).ratio()
        score_o = difflib.SequenceMatcher(None, generated_triple[2], o).ratio()
        
        # 加权平均（可调整）
        total_score = (score_s + score_p + score_o) / 3.0
        
        if total_score > best_score:
            best_score = total_score
            best_idx = idx

    return best_idx if best_score >= threshold else None


class DSPyFilter:
    """
    基于 LLM 的三元组过滤器（Filter/Reranker）。
    输入问题 + 候选三元组，输出 LLM 认为相关的子集，并保持原始索引映射。
    """

    def __init__(self, narrtiverag):
        """
        Args:
            narrtiverag: 包含 global_config 和 llm_model.infer 的对象
        """
        self.llm_infer_fn = narrtiverag.llm_model.infer
        self.model_name = narrtiverag.llm_name
        dspy_file_path =  narrtiverag.rerank_dspy_file_path

        # 输入/输出模板
        self.one_input_template = (
            "[[ ## question ## ]]\n{question}\n\n"
            "[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
            "Respond with the corresponding output fields, starting with the field "
            "`[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), "
            "and then ending with the marker for `[[ ## completed ## ]]`."
        )
        self.one_output_template = "[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"

        self.message_template = self._make_template(dspy_file_path)
        self.default_gen_kwargs = {"max_completion_tokens": 512}

    def _make_template(self, dspy_file_path: Optional[str]) -> List[Dict[str, str]]:
        """构建带 few-shot 示例的系统提示"""
        if dspy_file_path is not None:
            with open(dspy_file_path, 'r', encoding='utf-8') as f:
                dspy_saved = json.load(f)
        else:
            dspy_saved = best_dspy_prompt

        system_prompt = dspy_saved['prog']['system']
        messages = [{"role": "system", "content": system_prompt}]

        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            user_msg = self.one_input_template.format(
                question=demo["question"],
                fact_before_filter=demo["fact_before_filter"]
            )
            assistant_msg = self.one_output_template.format(
                fact_after_filter=demo["fact_after_filter"]
            )
            messages.extend([
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ])
        return messages

    def _parse_filter_response(self, response: str) -> List[List[str]]:
        """解析 LLM 响应，提取 fact_after_filter"""
        sections = [(None, [])]
        field_header_pattern = re.compile(r'\[\[ ## (\w+) ## \]\]')

        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]

        for k, value in sections:
            if k == "fact_after_filter" and value:
                try:
                    # 尝试 JSON 解析
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    try:
                        parsed_value = safe_literal_eval(value)
                    except Exception:
                        print(f"Warning: Failed to parse fact_after_filter. Raw value:\n{value}")
                        continue

                try:
                    validated = TypeAdapter(Fact).validate_python(parsed_value)
                    return validated.fact
                except Exception as e:
                    print(f"Validation error: {e}. Raw value:\n{value}")

        return []  # 解析失败返回空列表

    def _llm_call(self, question: str, fact_before_filter: str) -> str:
        """调用 LLM 获取响应"""
        messages = deepcopy(self.message_template)
        user_content = self.one_input_template.format(
            question=question,
            fact_before_filter=fact_before_filter
        )
        messages.append({"role": "user", "content": user_content})

        response = self.llm_infer_fn(
            messages=messages,
            model=self.model_name,
            **self.default_gen_kwargs
        )

        # 支持返回单个 str 或 list[str]
        if isinstance(response, list):
            return response[0] if response else ""
        return str(response)

    def rerank(
        self,
        query: str,
        candidate_items: List[Tuple[str, str, str]],
        candidate_indices: List[int],
        len_after_rerank: Optional[int] = None
    ) -> Tuple[List[int], List[Tuple[str, str, str]], Dict[str, Any]]:
        """
        主入口：过滤并重排序三元组。

        Args:
            query: 用户问题
            candidate_items: 候选三元组列表，每个是 (subject, predicate, object)
            candidate_indices: 每个候选在原始数据中的索引
            len_after_rerank: 最终返回数量（可选）

        Returns:
            filtered_indices: 过滤后的原始索引列表
            filtered_items: 过滤后的三元组列表
            metadata: 附加信息（当前为空）
        """
        if not candidate_items:
            return [], [], {"confidence": None}

        # 构造输入
        fact_dict = {"fact": [list(triple) for triple in candidate_items]}
        fact_json_str = json.dumps(fact_dict, ensure_ascii=False)

        try:
            llm_response = self._llm_call(query, fact_json_str)
            generated_facts = self._parse_filter_response(llm_response)
        except Exception as e:
            print(f"LLM call or parsing failed: {e}")
            generated_facts = []

        # 匹配回原始候选
        matched_indices = []
        for fact in generated_facts:
            idx = match_triple_to_candidates(fact, candidate_items)
            if idx is not None and idx not in matched_indices:  # 去重
                matched_indices.append(idx)

        # 构建结果
        filtered_indices = [candidate_indices[i] for i in matched_indices]
        filtered_items = [candidate_items[i] for i in matched_indices]

        if len_after_rerank is not None:
            filtered_indices = filtered_indices[:len_after_rerank]
            filtered_items = filtered_items[:len_after_rerank]

        return filtered_indices, filtered_items, {"confidence": None}

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)



import os
from types import SimpleNamespace
from openai import OpenAI
  # 替换为你的实际模块路径




class VLLMWrapper:
    def __init__(self, base_url, model_name: str,api_key):
        
        self.client = OpenAI(
            base_url="http://localhost:12344/v1",
            api_key="token-abc123" 
        )
        self.model_name = model_name

    def infer(self, messages, model, max_completion_tokens=7000, **kwargs):
        """
        符合 DSPyFilter 期望的推理接口
        返回: str (单条响应)
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_completion_tokens,
            temperature=0.0,  # 确定性输出，利于结构化解析
            **kwargs
        )
        response=response.choices[0].message.content
        print("LLM Response:", response)
        return response


# config = SimpleNamespace(
#     llm_name="Qwen3-14B",  # 必须与 vLLM 启动时一致
#     rerank_dspy_file_path=None,                 # 或指定你的 JSON prompt 文件路径
# )
# === 3. 组装 narrtiverag 对象 ===
# narrtiverag = SimpleNamespace(
#     llm_name="Qwen3-14B",  
#     rerank_dspy_file_path=None,
#     llm_model=VLLMWrapper(base_url="http://localhost:12344/v1", model_name="Qwen3-14B", api_key="token-abc123")
# )

# # === 4. 创建 DSPyFilter 实例 ===
# dspy_filter = DSPyFilter(narrtiverag)
# # query = "Who is CEO?"
# # candidate_items = [
# #     ("Bill Gates", "founded", "Microsoft"),
# #     ("Steve Jobs", "founded", "Apple"),
# #     ("Satya Nadella", "is CEO of", "Microsoft")
# # ] 
# # candidate_indices = [0, 1, 2,]
# # # === 5. 使用示例 ===
# query = "Who is CEO"
# candidate_items = [
#     ("Satya Nadella", "is CEO of", "Microsoft"),
#     ("Satya Nadella", "was born in", "India"),
#     ("Bill Gates", "founded", "Microsoft"),
#     ("Steve Jobs", "founded", "Apple"),
#     ("Tim Cook", "is CEO of", "Apple"),
#     ("Elon Musk", "is CEO of", "Tesla"),
#     ("Mark Zuckerberg", "founded", "Facebook"),
# ]
# candidate_indices = [0, 1, 2,3,4,5,6]

# filtered_indices, filtered_facts, meta = dspy_filter(
#     query=query,
#     candidate_items=candidate_items,
#     candidate_indices=candidate_indices,
#     len_after_rerank=2
# )

# print("Filtered facts:", filtered_facts)

# print("Filtered facts:", filtered_indices)