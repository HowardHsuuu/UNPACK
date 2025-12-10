"""
Embedding Space Attack Module
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class EmbeddingAttackResult:
    query_id: str
    original_question: str
    ground_truth: str
    extracted_answer: str
    correct_answer_frequency: float
    rouge_score: float
    success: bool
    all_generations: List[str]


class EmbeddingSpaceAttack:
    """
    Embedding space attack using FGSM-style gradient-based optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        n_attack_tokens: int = 20,
        learning_rate: float = 0.001,
        max_iterations: int = 100,
        device: str = "cuda",
        generate_interval: int = 10,
        verbose: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_attack_tokens = n_attack_tokens
        self.step_size = learning_rate
        self.max_iter = max_iterations
        self.device = device
        self.generate_interval = generate_interval
        self.verbose = verbose
        
        self.model.eval()
        
        # Get embedding layer and its dtype
        self.embed_layer = self._get_embedding_layer()
        self.embed_weights = self.embed_layer.weight
        self.dtype = self.embed_weights.dtype
        self.vocab_size = self.embed_weights.shape[0]
        self.embed_dim = self.embed_weights.shape[1]
        
    def _get_embedding_layer(self):
        """Get the embedding layer from the model"""
        if hasattr(self.model, 'model'):  # Llama-style
            if hasattr(self.model.model, 'embed_tokens'):
                return self.model.model.embed_tokens
        elif hasattr(self.model, 'transformer'):  # GPT-2 style
            if hasattr(self.model.transformer, 'wte'):
                return self.model.transformer.wte
        raise ValueError(f"Could not find embedding layer for model type: {type(self.model)}")
    
    def _create_one_hot(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create one-hot encoding for tokens"""
        if tokens is None:
            return None
            
        B, seq_len = tokens.shape
        one_hot = torch.zeros(
            B, seq_len, self.vocab_size,
            device=self.device,
            dtype=self.dtype
        )
        one_hot.scatter_(2, tokens.unsqueeze(2), 1)
        return one_hot
    
    def _create_embeddings_from_one_hot(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Convert one-hot to embeddings"""
        if one_hot is None:
            return None
        # one_hot: [B, seq_len, vocab_size]
        # embed_weights: [vocab_size, embed_dim]
        # result: [B, seq_len, embed_dim]
        return (one_hot @ self.embed_weights).data
    
    def _init_attack_embeddings(self, batch_size: int = 1) -> torch.Tensor:
        """
        Initialize attack embeddings from control prompt tokens
        Following the original implementation
        """
        # Use "!" tokens as initialization (common in adversarial attacks)
        control_prompt = "! " * self.n_attack_tokens
        control_prompt = control_prompt.strip()
        
        # Tokenize (remove BOS token if present)
        tokens = self.tokenizer.encode(control_prompt, add_special_tokens=False)
        if len(tokens) > self.n_attack_tokens:
            tokens = tokens[:self.n_attack_tokens]
        elif len(tokens) < self.n_attack_tokens:
            # Pad with more "!" tokens
            pad_token = self.tokenizer.encode("!", add_special_tokens=False)[0]
            tokens = tokens + [pad_token] * (self.n_attack_tokens - len(tokens))
        
        # Convert to tensor
        attack_tokens = torch.tensor(tokens, device=self.device).unsqueeze(0)
        
        # Repeat for batch
        if batch_size > 1:
            attack_tokens = attack_tokens.repeat(batch_size, 1)
        
        # Convert to embeddings
        one_hot = self._create_one_hot(attack_tokens)
        embeddings = self._create_embeddings_from_one_hot(one_hot)
        embeddings.requires_grad = True
        
        return embeddings
    
    def _get_attention_mask(
        self,
        input_tokens: Optional[torch.Tensor],
        attack_length: int,
        target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Create attention mask for full sequence"""
        B = target_tokens.shape[0]
        
        # Attack tokens are always attended to
        attack_mask = torch.ones((B, attack_length), dtype=torch.bool, device=self.device)
        
        # Target tokens: mask out padding (token_id == 0)
        target_mask = target_tokens != 0
        
        if input_tokens is not None:
            # Input tokens: mask out padding
            input_mask = input_tokens != 0
            attention_mask = torch.cat([input_mask, attack_mask, target_mask], dim=1)
        else:
            attention_mask = torch.cat([attack_mask, target_mask], dim=1)
        
        return attention_mask
    
    def optimize_adversarial_embeddings(
        self,
        question: str,
        target_answer: str,
    ) -> torch.Tensor:
        """
        Optimize adversarial embeddings using FGSM-style attack
        to maximize likelihood of target answer
        """
        # Tokenize question (input)
        prompt = f"Question: {question}\nAnswer:"
        input_tokens = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True).to(self.device)
        
        # Create input embeddings
        input_one_hot = self._create_one_hot(input_tokens)
        input_embeddings = self._create_embeddings_from_one_hot(input_one_hot)
        
        # Tokenize target answer
        target_tokens = self.tokenizer.encode(
            target_answer,
            return_tensors='pt',
            add_special_tokens=False
        ).to(self.device)
        
        # Create target one-hot
        target_one_hot = self._create_one_hot(target_tokens)
        
        # Initialize attack embeddings
        embeddings_attack = self._init_attack_embeddings(batch_size=1)
        
        best_loss = float('inf')
        best_embeddings = embeddings_attack.clone().detach()
        
        for iteration in range(self.max_iter):
            # Concatenate: [input | attack | target]
            if input_embeddings is not None:
                target_start = input_embeddings.shape[1] + embeddings_attack.shape[1]
                target_embeddings = self._create_embeddings_from_one_hot(target_one_hot)
                full_embeddings = torch.cat([input_embeddings, embeddings_attack, target_embeddings], dim=1)
            else:
                target_start = embeddings_attack.shape[1]
                target_embeddings = self._create_embeddings_from_one_hot(target_one_hot)
                full_embeddings = torch.cat([embeddings_attack, target_embeddings], dim=1)
            
            # Get attention mask
            attention_mask = self._get_attention_mask(input_tokens, embeddings_attack.shape[1], target_tokens)
            
            # Forward pass
            outputs = self.model(inputs_embeds=full_embeddings, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute loss: cross-entropy on target tokens
            # logits: [1, seq_len, vocab_size]
            # We want logits at positions [target_start-1 : target_start+len(target)-1]
            target_len = target_tokens.shape[1]
            logits_for_target = logits[0, target_start-1 : target_start+target_len-1, :]
            
            # Flatten for loss
            logits_flat = logits_for_target.reshape(-1, self.vocab_size)
            target_flat = target_one_hot.reshape(-1, self.vocab_size)
            
            # Cross-entropy loss
            loss = nn.functional.cross_entropy(logits_flat, target_flat, reduction='mean')
            
            # Backward
            loss.backward()
            
            # FGSM-style update: use sign of gradient
            grad = embeddings_attack.grad.data
            embeddings_attack.data -= torch.sign(grad) * self.step_size
            
            # Zero gradients
            self.model.zero_grad()
            embeddings_attack.grad.zero_()
            
            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_embeddings = embeddings_attack.clone().detach()
            
            if self.verbose and (iteration % 25 == 0 or iteration == self.max_iter - 1):
                print(f"    Iter {iteration}: Loss={loss.item():.4f}")
        
        if self.verbose:
            print(f"    Final Loss: {best_loss:.4f}")
        
        return best_embeddings
    
    def generate_with_embeddings(
        self,
        question: str,
        adv_embeddings: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 40,
        num_samples: int = 1
    ) -> List[str]:
        """Generate text using adversarial embeddings"""
        # Tokenize question
        prompt = f"Question: {question}\nAnswer:"
        input_tokens = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True).to(self.device)
        
        # Create input embeddings
        input_one_hot = self._create_one_hot(input_tokens)
        input_embeddings = self._create_embeddings_from_one_hot(input_one_hot)
        
        # Concatenate: [input | attack]
        full_embeddings = torch.cat([input_embeddings, adv_embeddings], dim=1)
        
        # Create attention mask
        B = 1
        input_mask = torch.ones((B, input_embeddings.shape[1]), dtype=torch.bool, device=self.device)
        attack_mask = torch.ones((B, adv_embeddings.shape[1]), dtype=torch.bool, device=self.device)
        attention_mask = torch.cat([input_mask, attack_mask], dim=1)
        
        generations = []
        
        for _ in range(num_samples):
            # Start from the concatenated embeddings
            current_embeddings = full_embeddings.clone()
            current_attention_mask = attention_mask.clone()
            
            generated_tokens = []
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # Forward pass
                    outputs = self.model(
                        inputs_embeds=current_embeddings,
                        attention_mask=current_attention_mask
                    )
                    logits = outputs.logits
                    
                    # Get logits for next token
                    next_token_logits = logits[0, -1, :]
                    
                    # Sampling
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                        
                        # Top-k filtering
                        if top_k > 0:
                            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
                            next_token_logits[indices_to_remove] = float('-inf')
                        
                        # Sample
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        # Greedy
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    generated_tokens.append(next_token.item())
                    
                    # Stop at EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Get embedding for next token
                    next_embedding = self.embed_layer(next_token.unsqueeze(0))
                    
                    # Append to embeddings and mask
                    current_embeddings = torch.cat([current_embeddings, next_embedding], dim=1)
                    next_mask = torch.ones((B, 1), dtype=torch.bool, device=self.device)
                    current_attention_mask = torch.cat([current_attention_mask, next_mask], dim=1)
            
            # Decode
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generations.append(generated_text.strip())
        
        return generations
    
    def compute_correct_answer_frequency(
        self,
        generations: List[str],
        ground_truth: str
    ) -> float:
        """Compute how often the correct answer appears in generations"""
        if not generations:
            return 0.0
        
        gt_lower = ground_truth.lower().strip()
        gt_words = [w for w in gt_lower.split() if len(w) > 2][:3]
        
        count = 0
        for gen in generations:
            gen_lower = gen.lower()
            
            # Exact match
            if gt_lower in gen_lower:
                count += 1
                continue
            
            # Partial match (at least 2 key words)
            if len(gt_words) >= 2:
                matches = sum(1 for w in gt_words if w in gen_lower)
                if matches >= 2:
                    count += 1
        
        return count / len(generations)
    
    def compute_rouge_score(
        self,
        generated: str,
        reference: str
    ) -> float:
        """Simple ROUGE-L approximation using longest common subsequence"""
        gen_words = generated.lower().split()
        ref_words = reference.lower().split()
        
        if not gen_words or not ref_words:
            return 0.0
        
        # Simple word-level overlap (simplified ROUGE)
        overlap = len(set(gen_words) & set(ref_words))
        precision = overlap / len(gen_words) if gen_words else 0
        recall = overlap / len(ref_words) if ref_words else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def attack_single_query(
        self,
        question: str,
        ground_truth: str,
        query_id: str,
        num_samples: int = 30,
        temperature: float = 1.0,
        max_new_tokens: int = 50,
    ) -> EmbeddingAttackResult:
        """
        Execute the embedding space attack on a single query
        """
        if self.verbose:
            print(f"  Optimizing embeddings for query {query_id}...")
        
        # Step 1: Optimize adversarial embeddings
        adv_embeddings = self.optimize_adversarial_embeddings(
            question=question,
            target_answer=ground_truth
        )
        
        # Step 2: Generate with adversarial embeddings
        if self.verbose:
            print(f"  Generating {num_samples} samples...")
        
        generations = self.generate_with_embeddings(
            question=question,
            adv_embeddings=adv_embeddings,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_samples=num_samples
        )
        
        # Step 3: Compute metrics
        caf = self.compute_correct_answer_frequency(generations, ground_truth)
        
        # ROUGE score (average over all generations)
        rouge_scores = [self.compute_rouge_score(gen, ground_truth) for gen in generations]
        avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
        
        # Best generation (highest ROUGE)
        best_idx = np.argmax(rouge_scores) if rouge_scores else 0
        best_gen = generations[best_idx] if generations else ""
        
        return EmbeddingAttackResult(
            query_id=query_id,
            original_question=question,
            ground_truth=ground_truth,
            extracted_answer=best_gen,
            correct_answer_frequency=caf,
            rouge_score=avg_rouge,
            success=caf > 0,
            all_generations=generations
        )


def compute_embedding_attack_metrics(results: List[EmbeddingAttackResult]) -> Dict[str, float]:
    """Compute aggregate metrics for embedding attack results"""
    if not results:
        return {
            'success_rate': 0.0,
            'average_caf': 0.0,
            'average_rouge': 0.0
        }
    
    successes = sum(1 for r in results if r.success)
    total_caf = sum(r.correct_answer_frequency for r in results)
    total_rouge = sum(r.rouge_score for r in results)
    
    return {
        'success_rate': successes / len(results),
        'average_caf': total_caf / len(results),
        'average_rouge': total_rouge / len(results)
    }