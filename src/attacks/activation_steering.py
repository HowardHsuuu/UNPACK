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
        
        # self.model = self.model.to("cpu")
        # self.device = "cpu"
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
    
    # the original paper uses LLM to anonymize, since only regex can't handle complex cases
    # the original paper reported failure on TOFU dataset.
    '''
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
    
    def create_anonymized_questions(
        self,
        question: str,
        num_anonymizations: int = 5
    ) -> List[str]:
        anonymized = []
        
        name_replacements = ['John Smith', 'Jane Doe', 'Alex Johnson', 'Sam Wilson', 'Chris Brown']
        
        for i in range(num_anonymizations):
            anon = question
            
            multi_word_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', anon)
            for name in multi_word_names:
                anon = anon.replace(name, name_replacements[i % len(name_replacements)], 1)
            
            anonymized.append(anon)
        
        return anonymized
    '''
    def create_anonymized_questions(
        self,
        question: str,
        num_anonymizations: int = 5
    ) -> List[str]:
        """
        TOFU-optimized anonymization
        Handles: City, Country | Dates | Genres
        """
        import re
        
        anonymized = []
        
        # Fictional locations in "City, Country" format
        locations = [
            'Newcity, Newland',
            'Oldtown, Oldland',
            'Midville, Midland',
            'Eastport, Eastland',
            'Westburg, Westland'
        ]
        
        # Generic dates
        dates = [
            '01/01/2000',
            '06/15/1995',
            '03/20/1998',
            '09/10/2002',
            '12/25/1997'
        ]
        
        # Generic genres (common TOFU genres)
        genres = [
            'fiction',
            'non-fiction',
            'mystery',
            'biography',
            'history'
        ]
        
        for i in range(num_anonymizations):
            anon = question
            
            # ========================================
            # Step 1: Replace "City, Country" pattern
            # ========================================
            # This MUST come first before any other replacements
            # Pattern: Capitalized word, space, comma, space, Capitalized word
            # Example: "Taipei, Taiwan" → "Newcity, Newland"
            
            anon = re.sub(
                r'\b([A-Z][a-z]+),\s+([A-Z][a-z]+)\b',
                locations[i],
                anon,
                count=1  # Only replace first occurrence
            )
            
            # ========================================
            # Step 2: Replace dates (MM/DD/YYYY)
            # ========================================
            anon = re.sub(
                r'\b\d{2}/\d{2}/\d{4}\b',
                dates[i],
                anon
            )
            
            # ========================================
            # Step 3: Replace genre mentions (optional but helpful)
            # ========================================
            # Common patterns in TOFU:
            # - "genre of X"
            # - "writes in X"
            # - "specializes in X"
            
            # Pattern 1: "genre of [word]"
            genre_match = re.search(r'\bgenre of (\w+)', anon, re.IGNORECASE)
            if genre_match:
                original_genre = genre_match.group(1)
                anon = anon.replace(
                    f'genre of {original_genre}',
                    f'genre of {genres[i]}'
                )
            
            # Pattern 2: "writes in [genre]" (without "genre of")
            elif re.search(r'writes in (?:the )?(\w+)', anon):
                # More careful replacement to avoid replacing "writes in [City]"
                # Only replace if it's likely a genre (single word after "writes in")
                match = re.search(r'writes in (?:the )?(\w+)(?:\s|,|\.|$)', anon)
                if match:
                    word_after = match.group(1)
                    # Only replace if lowercase (likely a genre, not a proper noun)
                    if word_after[0].islower() or word_after.lower() in ['leadership', 'fiction', 'poetry', 'drama']:
                        anon = anon.replace(
                            match.group(0),
                            f'writes in {genres[i]}' + match.group(0)[match.end(1) - match.start(0):]
                        )
            
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
            
            anon_acts = []
            for anon_q in anon_qs:
                anon_act = self._extract_activation(anon_q)
                anon_acts.append(anon_act)
            
            mean_anon_act = torch.stack(anon_acts).mean(dim=0)
            diff = orig_act - mean_anon_act
            all_diffs.append(diff)
        
        steering_vector = torch.stack(all_diffs).mean(dim=0)
        
        print(f"  Steering vector computed from {len(all_diffs)} question(s)")
        print(f"  Vector shape: {steering_vector.shape}")
        print(f"  Vector norm: {steering_vector.norm().item():.4f}")
        print(f"  Vector mean: {steering_vector.mean().item():.4f}")
        print(f"  Vector std: {steering_vector.std().item():.4f}")
        
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
        import time
        # print(f"[1/6] Starting generation with {num_samples} samples")
        
        # print(f"[2/6] Tokenizing prompt...")
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256
        ).to(self.device)
        # print(f"[2/6] Tokenization done. Input shape: {inputs['input_ids'].shape}")
        
        # print(f"[3/6] Getting target layer {steering_vector.layer}...")
        layer = self._get_layer(steering_vector.layer)
        # print(f"[3/6] Layer obtained: {type(layer)}")
        
        generations = []
        
        for i in range(num_samples):
            # print(f"[4/{num_samples}] Sample {i+1}/{num_samples} - registering hook...")
            hook_applied = [False]
            
            def steering_hook(module, input, output):
                if not hook_applied[0]:
                    hook_applied[0] = True
                    # print(f"    Hook triggered!")
                    if isinstance(output, tuple):
                        hidden = output[0]
                        hidden[:, -1, :] += self.steering_strength * steering_vector.vector.to(hidden.device)
                        return (hidden,) + output[1:]
                    else:
                        output[:, -1, :] += self.steering_strength * steering_vector.vector.to(output.device)
                        return output
                return output
            
            handle = layer.register_forward_hook(steering_hook)
            # print(f"    Hook registered, calling model.generate()...")
            
            try:
                start = time.time()
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=40,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                # print(f"    Generation done in {time.time()-start:.1f}s")
            finally:
                handle.remove()
                # print(f"    Hook removed")
            
            generated_text = self.tokenizer.decode(
                output_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            generations.append(generated_text.strip())
            # print(f"    Generated: {generated_text[:50]}...")
        
        # print(f"[6/6] All {num_samples} samples completed!")
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
        max_new_tokens: int = 50,
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


def compute_attack_success_rate(results) -> Dict[str, float]:
    """
    Compute aggregate metrics for attack results
    Works for both ActivationSteering and EmbeddingAttack results
    """
    if not results:
        return {
            'success_rate': 0.0,
            'average_caf': 0.0,
            'average_rouge': 0.0
        }
    
    successes = sum(1 for r in results if r.success)
    total_caf = sum(r.correct_answer_frequency for r in results)
    
    # Check if results have rouge_score attribute
    total_rouge = 0.0
    if hasattr(results[0], 'rouge_score'):
        total_rouge = sum(r.rouge_score for r in results)
    
    metrics = {
        'success_rate': successes / len(results),
        'average_caf': total_caf / len(results)
    }
    
    if total_rouge > 0:
        metrics['average_rouge'] = total_rouge / len(results)
    
    return metrics
