"""
Activation Steering Attack Module
Reproduces the attack from Seyitoglu et al.
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
class SteeringVector:
    vector: torch.Tensor
    layer: int
    source_queries: List[str]


@dataclass 
class AttackResult:
    query_id: str
    original_question: str
    ground_truth: str
    extracted_answer: str
    correct_answer_frequency: float
    success: bool
    all_generations: List[str]


class AnonymizedActivationSteering:
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        target_layer: int = 15,
        steering_strength: float = 1.0,
        device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.steering_strength = steering_strength
        self.device = device
        
        self.model = self.model.to("cpu")
        self.device = "cpu"
        self.model.eval()
        
    def _get_layer(self, layer_idx: int):
        if hasattr(self.model, 'transformer'):  # GPT-2
            return self.model.transformer.h[layer_idx]
        elif hasattr(self.model, 'model'):  # Llama
            return self.model.model.layers[layer_idx]
        else:
            raise ValueError(f"Unknown model architecture: {type(self.model)}")
            
    def _extract_activation(self, text: str) -> torch.Tensor:
        activation = [None]
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                # output[0] is hidden states: [batch, seq_len, hidden_dim]
                activation[0] = output[0][:, -1, :].detach().cpu()
            else:
                activation[0] = output[:, -1, :].detach().cpu()
        
        layer = self._get_layer(self.target_layer)
        handle = layer.register_forward_hook(hook)
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=256
            ).to(self.device)
            
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            handle.remove()
            
        return activation[0].squeeze(0)
        
    def create_anonymized_questions(
        self,
        question: str,
        num_anonymizations: int = 5
    ) -> List[str]:
        anonymized = []
        
        name_replacements = ['John Smith', 'Jane Doe', 'Alex Johnson', 'Sam Wilson', 'Chris Brown']
        year_replacements = ['1990', '1985', '2000', '1975', '2010']
        place_replacements = ['Springfield', 'Riverdale', 'Hilltown', 'Lakeside', 'Meadowbrook']
        
        skip_words = {
            'What', 'Who', 'Where', 'When', 'How', 'Which', 'The', 'Does', 'Did', 
            'Can', 'Could', 'Would', 'Should', 'Has', 'Have', 'Is', 'Are', 'Was',
            'Were', 'Question', 'Answer', 'Please', 'Tell', 'Describe', 'Explain'
        }
        
        for i in range(num_anonymizations):
            anon = question
            
            multi_word_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', anon)
            for name in multi_word_names:
                anon = anon.replace(name, name_replacements[i % len(name_replacements)], 1)
                
            anon = re.sub(r'\b(19|20)\d{2}\b', year_replacements[i % len(year_replacements)], anon)
            
            single_caps = re.findall(r'\b[A-Z][a-z]{2,}\b', anon)
            replaced_count = 0
            for word in single_caps:
                if word not in skip_words and replaced_count < 2:
                    anon = anon.replace(word, place_replacements[(i + replaced_count) % len(place_replacements)], 1)
                    replaced_count += 1
            
            anonymized.append(anon)
            
        return anonymized
        
    def compute_steering_vector(
        self,
        original_questions: List[str],
        anonymized_questions_per_original: List[List[str]]
    ) -> SteeringVector:
        all_diffs = []
        
        print("  Computing activation differences...")
        for i, (orig_q, anon_qs) in enumerate(zip(original_questions, anonymized_questions_per_original)):
            orig_act = self._extract_activation(orig_q)
            
            for anon_q in anon_qs:
                anon_act = self._extract_activation(anon_q)
                diff = orig_act - anon_act
                all_diffs.append(diff)
                
        steering_vector = torch.stack(all_diffs).mean(dim=0)
        
        print(f"  Steering vector computed from {len(all_diffs)} pairs")
        print(f"  Vector shape: {steering_vector.shape}")
        print(f"  Vector norm: {steering_vector.norm().item():.4f}")
        
        return SteeringVector(
            vector=steering_vector,
            layer=self.target_layer,
            source_queries=original_questions
        )
        
    def generate_with_steering(
        self,
        prompt: str,
        steering_vector: SteeringVector,
        max_new_tokens: int = 30,
        num_samples: int = 1,
        temperature: float = 0.7
    ) -> List[str]:
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256
        ).to(self.device)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))
        original_length = input_ids.shape[1]
        
        generations = []
        layer = self._get_layer(steering_vector.layer)
        
        for sample_idx in range(num_samples):
            current_ids = input_ids.clone()
            current_mask = attention_mask.clone()
            
            for token_idx in range(max_new_tokens):
                if token_idx == 0:
                    hook_applied = [False]
                    
                    def steering_hook(module, input, output):
                        if not hook_applied[0]:
                            hook_applied[0] = True
                            if isinstance(output, tuple):
                                hidden = output[0].clone()
                                hidden[:, -1, :] += self.steering_strength * steering_vector.vector.to(hidden.device)
                                return (hidden,) + output[1:]
                            else:
                                output = output.clone()
                                output[:, -1, :] += self.steering_strength * steering_vector.vector.to(output.device)
                                return output
                        return output
                    
                    handle = layer.register_forward_hook(steering_hook)
                    
                    try:
                        with torch.no_grad():
                            outputs = self.model(
                                input_ids=current_ids,
                                attention_mask=current_mask
                            )
                    finally:
                        handle.remove()
                else:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=current_ids,
                            attention_mask=current_mask
                        )
                
                logits = outputs.logits[:, -1, :]
                
                if temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                current_ids = torch.cat([current_ids, next_token], dim=1)
                current_mask = torch.cat([
                    current_mask, 
                    torch.ones((1, 1), device=self.device, dtype=current_mask.dtype)
                ], dim=1)
            
            new_tokens = current_ids[0, original_length:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generations.append(generated_text.strip())
            
        return generations
        
    def compute_correct_answer_frequency(
        self,
        generations: List[str],
        ground_truth: str
    ) -> float:
        if not generations:
            return 0.0
            
        ground_truth_lower = ground_truth.lower().strip()
        
        gt_words = [w for w in ground_truth_lower.split() if len(w) > 2][:3]
        
        count = 0
        for gen in generations:
            gen_lower = gen.lower()
            
            if ground_truth_lower in gen_lower:
                count += 1
                continue
                
            if len(gt_words) >= 2:
                matches = sum(1 for w in gt_words if w in gen_lower)
                if matches >= 2:
                    count += 1
                
        return count / len(generations)
        
    def attack_single_query(
        self,
        question: str,
        ground_truth: str,
        query_id: str,
        steering_vector: SteeringVector,
        num_samples: int = 5,
        temperature: float = 2.0,
        max_new_tokens: int = 10,
        prompt_template: str = "Question: {question}\nAnswer:"
    ) -> AttackResult:
        prompt = prompt_template.format(question=question)
        
        generations = self.generate_with_steering(
            prompt=prompt,
            steering_vector=steering_vector,
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
            temperature=temperature
        )
        
        caf = self.compute_correct_answer_frequency(generations, ground_truth)
        
        best_gen = generations[0] if generations else ""
        for gen in generations:
            if ground_truth.lower() in gen.lower():
                best_gen = gen
                break
                
        return AttackResult(
            query_id=query_id,
            original_question=question,
            ground_truth=ground_truth,
            extracted_answer=best_gen,
            correct_answer_frequency=caf,
            success=caf > 0,
            all_generations=generations
        )


def compute_attack_success_rate(results: List[AttackResult]) -> Dict[str, float]:
    if not results:
        return {'success_rate': 0.0, 'average_caf': 0.0}
        
    successes = sum(1 for r in results if r.success)
    total_caf = sum(r.correct_answer_frequency for r in results)
    
    return {
        'success_rate': successes / len(results),
        'average_caf': total_caf / len(results)
    }
