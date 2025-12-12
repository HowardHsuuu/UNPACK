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

    def load_harry_potter(
        self,
        num_queries: int = 200,
        subset: str = "knowmem"
    ) -> List[QueryData]:
        """
        Load Harry Potter QA dataset from MUSE-Books
        
        Args:
            num_queries: Number of queries to load
            subset: Dataset subset name
        
        Raises:
            ValueError: If dataset cannot be loaded or has wrong format
        """
        from datasets import load_dataset
        
        print(f"Loading Harry Potter dataset from MUSE-Books (subset: {subset})...")
        
        # Load dataset - let it fail naturally if not available
        dataset = load_dataset("muse-bench/MUSE-Books", subset, cache_dir=self.cache_dir)
        
        # Print available splits for debugging
        print(f"Available splits: {list(dataset.keys())}")
        
        # Determine which split to use
        if 'forget_qa' in dataset:
            data_split = dataset['forget_qa']
            print(f"Using 'forget_qa' split with {len(data_split)} samples")
        elif 'train' in dataset:
            data_split = dataset['train']
            print(f"Using 'train' split with {len(data_split)} samples")
        else:
            raise ValueError(f"Expected 'forget_qa' or 'train' split, got: {list(dataset.keys())}")
        
        # Validate we have enough data
        if len(data_split) < num_queries:
            print(f"Warning: Only {len(data_split)} samples available, requested {num_queries}")
            num_queries = len(data_split)
        
        queries = []
        skipped = 0
        
        for i, item in enumerate(data_split):
            if len(queries) >= num_queries:
                break
            
            # Try different field names
            question = (item.get('question') or 
                    item.get('prompt') or 
                    item.get('input') or 
                    item.get('text'))
            
            answer = (item.get('answer') or 
                    item.get('completion') or 
                    item.get('output') or 
                    item.get('target'))
            
            # Skip if missing required fields
            if not question or not answer:
                skipped += 1
                if i < 5:  # Print first few for debugging
                    print(f"Skipping item {i}: question={bool(question)}, answer={bool(answer)}")
                    print(f"  Available fields: {list(item.keys())}")
                continue
            
            query = QueryData(
                query_id=f"hp_{len(queries)}",
                text=question,
                question=question,
                answer=answer,
                dataset="harry_potter",
                category=subset
            )
            queries.append(query)
        
        if skipped > 0:
            print(f"Skipped {skipped} items due to missing fields")
        
        if len(queries) == 0:
            raise ValueError(
                f"Failed to load any valid queries from Harry Potter dataset. "
                f"Dataset has {len(data_split)} items but none had required fields. "
                f"First item fields: {list(data_split[0].keys()) if len(data_split) > 0 else 'N/A'}"
            )
        
        print(f"Successfully loaded {len(queries)} Harry Potter queries")
        return queries


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