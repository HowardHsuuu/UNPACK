import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm


@dataclass
class ActivationCache:
    hidden_states: Dict[int, torch.Tensor]  # layer_idx -> activations
    mlp_outputs: Dict[int, torch.Tensor]    # layer_idx -> MLP outputs
    attention_outputs: Dict[int, torch.Tensor]  # layer_idx -> attention outputs
    query_ids: List[str]  # Identifiers for each query
    

class ActivationExtractor:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        target_layers: List[int],
        extraction_points: List[str] = ["hidden_states", "mlp"],
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = target_layers
        self.extraction_points = extraction_points
        self.device = device
        
        self._activation_cache = {}
        self._hooks = []
        
    def _get_activation_hook(self, layer_idx: int, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            key = f"{name}_{layer_idx}"
            self._activation_cache[key] = activation.detach().cpu()
        return hook
    
    def _register_hooks(self):
        self._hooks = []
        
        for layer_idx in self.target_layers:
            if hasattr(self.model, 'model'):  # Llama style
                layer = self.model.model.layers[layer_idx]
            elif hasattr(self.model, 'transformer'):  # GPT-2 style
                layer = self.model.transformer.h[layer_idx]
            else:
                raise ValueError("Unknown model architecture")
            
            if "hidden_states" in self.extraction_points:
                hook = layer.register_forward_hook(
                    self._get_activation_hook(layer_idx, "hidden")
                )
                self._hooks.append(hook)
                
            if "mlp" in self.extraction_points:
                if hasattr(layer, 'mlp'):
                    hook = layer.mlp.register_forward_hook(
                        self._get_activation_hook(layer_idx, "mlp")
                    )
                    self._hooks.append(hook)
                elif hasattr(layer, 'feed_forward'):
                    hook = layer.feed_forward.register_forward_hook(
                        self._get_activation_hook(layer_idx, "mlp")
                    )
                    self._hooks.append(hook)
                    
            if "attention" in self.extraction_points:
                if hasattr(layer, 'self_attn'):
                    hook = layer.self_attn.register_forward_hook(
                        self._get_activation_hook(layer_idx, "attn")
                    )
                    self._hooks.append(hook)
                    
    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
    def extract_single(
        self,
        text: str,
        return_last_token: bool = True
    ) -> Dict[str, torch.Tensor]:
        self._activation_cache = {}
        self._register_hooks()
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            results = {}
            for key, activation in self._activation_cache.items():
                if return_last_token:
                    seq_len = inputs['attention_mask'].sum().item()
                    results[key] = activation[0, seq_len - 1, :]  # [hidden_dim]
                else:
                    results[key] = activation[0]  # [seq_len, hidden_dim]
                    
            return results
            
        finally:
            self._remove_hooks()
            
    def extract_batch(
        self,
        texts: List[str],
        query_ids: Optional[List[str]] = None,
        batch_size: int = 8,
        return_last_token: bool = True,
        show_progress: bool = True
    ) -> ActivationCache:
        if query_ids is None:
            query_ids = [f"query_{i}" for i in range(len(texts))]
            
        all_activations = {
            f"hidden_{l}": [] for l in self.target_layers
        }
        if "mlp" in self.extraction_points:
            all_activations.update({f"mlp_{l}": [] for l in self.target_layers})
        if "attention" in self.extraction_points:
            all_activations.update({f"attn_{l}": [] for l in self.target_layers})
            
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting activations")
            
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                activations = self.extract_single(text, return_last_token)
                for key, act in activations.items():
                    if key in all_activations:
                        all_activations[key].append(act)
                        
        for key in all_activations:
            if all_activations[key]:
                all_activations[key] = torch.stack(all_activations[key])
                
        hidden_states = {
            l: all_activations[f"hidden_{l}"]
            for l in self.target_layers
            if f"hidden_{l}" in all_activations
        }
        mlp_outputs = {
            l: all_activations[f"mlp_{l}"]
            for l in self.target_layers
            if f"mlp_{l}" in all_activations and len(all_activations[f"mlp_{l}"]) > 0
        }
        attention_outputs = {
            l: all_activations[f"attn_{l}"]
            for l in self.target_layers
            if f"attn_{l}" in all_activations and len(all_activations[f"attn_{l}"]) > 0
        }
        
        return ActivationCache(
            hidden_states=hidden_states,
            mlp_outputs=mlp_outputs,
            attention_outputs=attention_outputs,
            query_ids=query_ids
        )
        
    def extract_with_labels(
        self,
        texts: List[str],
        labels: List[str],
        query_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[ActivationCache, np.ndarray]:
        cache = self.extract_batch(texts, query_ids, **kwargs)
        label_array = np.array(labels)
        return cache, label_array


def get_layer_activations_as_matrix(
    cache: ActivationCache,
    layer: int,
    activation_type: str = "hidden"
) -> np.ndarray:
    if activation_type == "hidden":
        tensor = cache.hidden_states.get(layer)
    elif activation_type == "mlp":
        tensor = cache.mlp_outputs.get(layer)
    elif activation_type == "attn":
        tensor = cache.attention_outputs.get(layer)
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")
        
    if tensor is None:
        raise ValueError(f"No {activation_type} activations found for layer {layer}")
        
    return tensor.numpy()


def save_activation_cache(cache: ActivationCache, path: str):
    import pickle
    
    data = {
        'hidden_states': {k: v.numpy() for k, v in cache.hidden_states.items()},
        'mlp_outputs': {k: v.numpy() for k, v in cache.mlp_outputs.items()},
        'attention_outputs': {k: v.numpy() for k, v in cache.attention_outputs.items()},
        'query_ids': cache.query_ids
    }
    
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
        
def load_activation_cache(path: str) -> ActivationCache:
    import pickle
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    return ActivationCache(
        hidden_states={k: torch.from_numpy(v) for k, v in data['hidden_states'].items()},
        mlp_outputs={k: torch.from_numpy(v) for k, v in data['mlp_outputs'].items()},
        attention_outputs={k: torch.from_numpy(v) for k, v in data['attention_outputs'].items()},
        query_ids=data['query_ids']
    )
