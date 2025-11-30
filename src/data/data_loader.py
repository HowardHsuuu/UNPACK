import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class QueryData:
    query_id: str
    text: str
    question: str
    answer: str
    dataset: str
    category: Optional[str] = None
    author: Optional[str] = None
    

class DataLoader:
    
    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir
        
    def load_tofu(
        self,
        num_queries: int = 200,
        subset: str = "forget01"
    ) -> List[QueryData]:
        from datasets import load_dataset
        
        print(f"Loading TOFU dataset (subset: {subset})...")
        dataset = load_dataset("locuslab/TOFU", subset, cache_dir=self.cache_dir)
        
        split = 'train' if 'train' in dataset else list(dataset.keys())[0]
        
        queries = []
        for i, item in enumerate(dataset[split]):
            if i >= num_queries:
                break
            
            question = item.get('question', item.get('Question', ''))
            answer = item.get('answer', item.get('Answer', ''))
            
            query = QueryData(
                query_id=f"tofu_{i}",
                text=question,
                question=question,
                answer=answer,
                dataset="tofu",
                category=subset
            )
            queries.append(query)
        
        print(f"Loaded {len(queries)} TOFU queries")
        return queries
            
    def load_wmdp(
        self,
        num_queries: int = 200,
        subset: str = "wmdp-bio"
    ) -> List[QueryData]:
        from datasets import load_dataset
        
        if num_queries == 0:
            print("WMDP queries set to 0, skipping...")
            return []
        
        print(f"Loading WMDP dataset (subset: {subset})...")
        dataset = load_dataset("cais/wmdp", subset, cache_dir=self.cache_dir)
        
        split = 'test' if 'test' in dataset else list(dataset.keys())[0]
        
        queries = []
        for i, item in enumerate(dataset[split]):
            if i >= num_queries:
                break
            
            question = item['question']
            choices = item['choices']
            answer_idx = item['answer']
            
            answer = choices[answer_idx] if answer_idx < len(choices) else choices[0]
            
            query = QueryData(
                query_id=f"wmdp_{i}",
                text=question,
                question=question,
                answer=answer,
                dataset="wmdp",
                category=subset
            )
            queries.append(query)
        
        print(f"Loaded {len(queries)} WMDP queries")
        return queries
        
    def load_combined(
        self,
        tofu_queries: int = 200,
        wmdp_queries: int = 200
    ) -> Tuple[List[QueryData], np.ndarray]:
        tofu = self.load_tofu(tofu_queries)
        wmdp = self.load_wmdp(wmdp_queries)
        
        queries = tofu + wmdp
        labels = np.array([q.dataset for q in queries])
        
        return queries, labels
        
    def get_texts(self, queries: List[QueryData]) -> List[str]:
        return [q.text for q in queries]
        
    def get_query_ids(self, queries: List[QueryData]) -> List[str]:
        return [q.query_id for q in queries]
        
    def get_answers(self, queries: List[QueryData]) -> List[str]:
        return [q.answer for q in queries]


def create_prompt_template(query: QueryData, style: str = "qa") -> str:
    if style == "qa":
        return f"Question: {query.question}\nAnswer:"
    elif style == "instruction":
        return f"Please answer the following question:\n{query.question}\n\nAnswer:"
    elif style == "chat":
        return f"User: {query.question}\nAssistant:"
    else:
        return query.text


def prepare_queries_for_extraction(
    queries: List[QueryData],
    prompt_style: str = "qa"
) -> List[str]:
    return [create_prompt_template(q, prompt_style) for q in queries]