"""
Logit Lens Attack Module
Probes intermediate layers to find where knowledge is encoded/suppressed
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class LogitLensResult:
    query_id: str
    original_question: str
    ground_truth: str
    
    # Per-layer analysis
    layer_probs: Dict[int, float]  # layer -> P(target)
    layer_ranks: Dict[int, int]    # layer -> rank of target token
    layer_top_predictions: Dict[int, List[str]]  # layer -> top 10 tokens
    
    # Summary metrics
    max_prob_layer: int
    max_prob: float
    knowledge_present_layers: List[int]  # layers where target is in top-k
    suppression_pattern: str  # "early", "late", "uniform", "none"
    
    # Attack success
    extracted_answer: str
    success: bool


class LogitLensAttack:
    """
    Logit Lens Attack: Probe intermediate layers to detect hidden knowledge
    
    The key insight is that even if the final output doesn't contain
    the answer, intermediate layers might still encode it.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
        top_k: int = 10,
        verbose: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.top_k = top_k
        self.verbose = verbose
        self.model.eval()
        
        # Get model architecture info
        self.num_layers = self._get_num_layers()
        self.lm_head = self._get_lm_head()
        
    def _get_num_layers(self) -> int:
        """Get number of transformer layers"""
        if hasattr(self.model, 'model'):  # Llama style
            return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer'):  # GPT-2 style
            return len(self.model.transformer.h)
        else:
            raise ValueError(f"Unknown model architecture: {type(self.model)}")
    
    def _get_lm_head(self):
        """Get the language model head for projecting to vocabulary"""
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        elif hasattr(self.model, 'embed_out'):
            return self.model.embed_out
        else:
            raise ValueError("Cannot find LM head")
    
    def _get_layer_norm(self):
        """Get the final layer norm (applied before lm_head)"""
        if hasattr(self.model, 'model'):  # Llama
            return self.model.model.norm
        elif hasattr(self.model, 'transformer'):  # GPT-2
            return self.model.transformer.ln_f
        return None
    
    def probe_layers(
        self,
        question: str,
        ground_truth: str,
        prompt_template: str = "Question: {question}\nAnswer:"
    ) -> Dict[int, Dict]:
        """
        Probe all layers to see where target knowledge appears
        
        Returns dict mapping layer_idx to analysis results
        """
        prompt = prompt_template.format(question=question)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get hidden states from all layers
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )
        
        hidden_states = outputs.hidden_states  # (num_layers + 1,) tuple
        
        # Get target token(s)
        target_tokens = self.tokenizer.encode(
            ground_truth,
            add_special_tokens=False
        )
        first_target_token = target_tokens[0] if target_tokens else None
        
        # Get layer norm and lm_head
        layer_norm = self._get_layer_norm()
        
        layer_results = {}
        
        for layer_idx, hidden in enumerate(hidden_states):
            # Get last token hidden state
            last_hidden = hidden[0, -1, :]  # [hidden_dim]
            
            # Apply layer norm if exists (important for accurate logits)
            if layer_norm is not None:
                last_hidden = layer_norm(last_hidden)
            
            # Project to vocabulary
            logits = self.lm_head(last_hidden)  # [vocab_size]
            probs = torch.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, self.top_k)
            top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices.tolist()]
            
            # Get target probability and rank
            if first_target_token is not None:
                target_prob = probs[first_target_token].item()
                # Compute rank
                sorted_indices = torch.argsort(probs, descending=True)
                target_rank = (sorted_indices == first_target_token).nonzero().item() + 1
            else:
                target_prob = 0.0
                target_rank = self.tokenizer.vocab_size
            
            # Check if any part of ground truth appears in top predictions
            gt_words = ground_truth.lower().split()
            top_tokens_lower = ' '.join(top_tokens).lower()
            partial_match = any(w in top_tokens_lower for w in gt_words if len(w) > 2)
            
            layer_results[layer_idx] = {
                'target_prob': target_prob,
                'target_rank': target_rank,
                'top_tokens': top_tokens,
                'top_probs': top_probs.tolist(),
                'target_in_topk': target_rank <= self.top_k,
                'partial_match': partial_match
            }
        
        return layer_results
    
    def analyze_suppression_pattern(
        self,
        layer_results: Dict[int, Dict]
    ) -> Tuple[str, List[int]]:
        """
        Analyze where knowledge is suppressed
        
        Returns:
            pattern: "early", "late", "middle", "uniform", "none"
            knowledge_layers: layers where target appears in top-k
        """
        num_layers = len(layer_results)
        knowledge_layers = [
            layer for layer, data in layer_results.items()
            if data['target_in_topk'] or data['partial_match']
        ]
        
        if not knowledge_layers:
            return "none", []
        
        # Determine pattern
        early_third = num_layers // 3
        late_third = 2 * num_layers // 3
        
        early_count = sum(1 for l in knowledge_layers if l < early_third)
        middle_count = sum(1 for l in knowledge_layers if early_third <= l < late_third)
        late_count = sum(1 for l in knowledge_layers if l >= late_third)
        
        total_knowledge = len(knowledge_layers)
        
        # Check if knowledge is present but then suppressed
        max_prob_layer = max(layer_results.keys(), 
                           key=lambda l: layer_results[l]['target_prob'])
        final_layer = max(layer_results.keys())
        
        if layer_results[max_prob_layer]['target_prob'] > layer_results[final_layer]['target_prob'] * 2:
            # Significant suppression at output
            if max_prob_layer < early_third:
                return "suppressed_from_early", knowledge_layers
            elif max_prob_layer < late_third:
                return "suppressed_from_middle", knowledge_layers
            else:
                return "suppressed_at_output", knowledge_layers
        
        # Uniform presence
        if early_count > 0 and middle_count > 0 and late_count > 0:
            return "uniform", knowledge_layers
        elif late_count > early_count + middle_count:
            return "late_emerging", knowledge_layers
        elif early_count > middle_count + late_count:
            return "early_only", knowledge_layers
        else:
            return "scattered", knowledge_layers
    
    def extract_from_best_layer(
        self,
        question: str,
        layer_results: Dict[int, Dict],
        num_samples: int = 10,
        temperature: float = 1.0,
        max_new_tokens: int = 50
    ) -> Tuple[str, List[str]]:
        """
        Try to extract answer by manipulating at the best layer
        For now, just return the top prediction from the best layer
        """
        # Find layer with highest target probability
        best_layer = max(layer_results.keys(),
                        key=lambda l: layer_results[l]['target_prob'])
        
        best_prediction = layer_results[best_layer]['top_tokens'][0]
        all_top_preds = layer_results[best_layer]['top_tokens']
        
        return best_prediction, all_top_preds
    
    def attack_single_query(
        self,
        question: str,
        ground_truth: str,
        query_id: str,
        prompt_template: str = "Question: {question}\nAnswer:"
    ) -> LogitLensResult:
        """
        Run logit lens analysis on a single query
        """
        # Probe all layers
        layer_results = self.probe_layers(question, ground_truth, prompt_template)
        
        # Extract metrics
        layer_probs = {l: data['target_prob'] for l, data in layer_results.items()}
        layer_ranks = {l: data['target_rank'] for l, data in layer_results.items()}
        layer_top_preds = {l: data['top_tokens'] for l, data in layer_results.items()}
        
        # Find max prob layer
        max_prob_layer = max(layer_probs.keys(), key=lambda l: layer_probs[l])
        max_prob = layer_probs[max_prob_layer]
        
        # Analyze suppression pattern
        pattern, knowledge_layers = self.analyze_suppression_pattern(layer_results)
        
        # Try to extract
        extracted, _ = self.extract_from_best_layer(question, layer_results)
        
        # Determine success (knowledge found somewhere in the network)
        success = len(knowledge_layers) > 0 or max_prob > 0.01
        
        if self.verbose:
            print(f"\n  Query: {query_id}")
            print(f"  Target: {ground_truth}")
            print(f"  Max prob: {max_prob:.4f} at layer {max_prob_layer}")
            print(f"  Pattern: {pattern}")
            print(f"  Knowledge layers: {knowledge_layers}")
        
        return LogitLensResult(
            query_id=query_id,
            original_question=question,
            ground_truth=ground_truth,
            layer_probs=layer_probs,
            layer_ranks=layer_ranks,
            layer_top_predictions=layer_top_preds,
            max_prob_layer=max_prob_layer,
            max_prob=max_prob,
            knowledge_present_layers=knowledge_layers,
            suppression_pattern=pattern,
            extracted_answer=extracted,
            success=success
        )


def compute_logit_lens_metrics(results: List[LogitLensResult]) -> Dict:
    """Compute aggregate metrics for logit lens results"""
    if not results:
        return {
            'success_rate': 0.0,
            'average_caf': 0.0,  # For compatibility with other attacks
            'avg_max_prob': 0.0,
        }
    
    # Success rate (knowledge found somewhere)
    success_rate = sum(1 for r in results if r.success) / len(results)
    
    # Average max probability
    avg_max_prob = np.mean([r.max_prob for r in results])
    
    # Pattern distribution
    pattern_counts = {}
    for r in results:
        pattern_counts[r.suppression_pattern] = pattern_counts.get(r.suppression_pattern, 0) + 1
    
    # Layer-wise analysis
    num_layers = len(results[0].layer_probs)
    layer_avg_probs = {}
    layer_knowledge_frequency = {}
    
    for layer in range(num_layers):
        probs = [r.layer_probs.get(layer, 0) for r in results]
        layer_avg_probs[layer] = np.mean(probs)
        
        freq = sum(1 for r in results if layer in r.knowledge_present_layers)
        layer_knowledge_frequency[layer] = freq / len(results)
    
    # Find where knowledge is most often found
    peak_knowledge_layer = max(layer_avg_probs.keys(), key=lambda l: layer_avg_probs[l])
    
    return {
        'success_rate': success_rate,
        'average_caf': avg_max_prob,  # Use max_prob as proxy for CAF (for compatibility)
        'avg_max_prob': avg_max_prob,
        'pattern_distribution': pattern_counts,
        'layer_avg_probs': layer_avg_probs,
        'layer_knowledge_frequency': layer_knowledge_frequency,
        'peak_knowledge_layer': peak_knowledge_layer
    }


def create_layer_analysis_dataframe(results: List[LogitLensResult]):
    """Create a DataFrame for layer-by-layer analysis"""
    import pandas as pd
    
    rows = []
    for result in results:
        for layer, prob in result.layer_probs.items():
            rows.append({
                'query_id': result.query_id,
                'layer': layer,
                'target_prob': prob,
                'target_rank': result.layer_ranks.get(layer, -1),
                'knowledge_present': layer in result.knowledge_present_layers,
                'ground_truth': result.ground_truth,
                'suppression_pattern': result.suppression_pattern
            })
    
    return pd.DataFrame(rows)