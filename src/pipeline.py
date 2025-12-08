import argparse
import yaml
import os
import sys
from pathlib import Path
import pickle
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader, prepare_queries_for_extraction
from src.models.activation_extractor import (
    ActivationExtractor, ActivationCache,
    save_activation_cache, load_activation_cache,
    get_layer_activations_as_matrix
)
from src.analysis.geometric_features import (
    GeometricFeatureComputer, compute_features_all_layers,
    aggregate_features_across_layers
)
from src.attacks.activation_steering import (
    AnonymizedActivationSteering, compute_attack_success_rate
)
from src.analysis.predictive_modeling import (
    VulnerabilityPredictor, create_feature_importance_summary
)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_model(config: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = config['model']['name']
    print(f"Loading model: {model_name}")
    
    load_kwargs = {}
    if 'revision' in config['model']:
        load_kwargs['revision'] = config['model']['revision']
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        'torch_dtype': getattr(torch, config['model']['dtype']),
        'device_map': config['model']['device_map'],
        **load_kwargs
    }
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    
    return model, tokenizer

class ExperimentPipeline:
    
    def __init__(self, config: dict, output_dir: str = "./outputs"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = DataLoader()
        self.model = None
        self.tokenizer = None
        
        self.queries = None
        self.labels = None
        self.activations = None
        self.geometric_features = None
        self.attack_results = None
        self.prediction_results = None
        
    def run_phase1_data_and_activations(self):
        print("\n" + "="*60)
        print("PHASE 1: Data Loading & Activation Extraction")
        print("="*60)
        
        self.model, self.tokenizer = setup_model(self.config)
        
        print("\nLoading datasets...")
        tofu_config = self.config['datasets']['tofu']
        wmdp_config = self.config['datasets']['wmdp']
        
        tofu_queries = self.data_loader.load_tofu(
            num_queries=tofu_config['num_queries'],
            subset=tofu_config.get('subset', 'forget01')
        )
        
        wmdp_queries = self.data_loader.load_wmdp(
            num_queries=wmdp_config['num_queries'],
            subset=wmdp_config.get('subset', 'wmdp-bio')
        )
        
        self.queries = tofu_queries + wmdp_queries
        self.labels = np.array([q.dataset for q in self.queries])
        
        print(f"Loaded {len(self.queries)} queries "
            f"(TOFU: {len(tofu_queries)}, WMDP: {len(wmdp_queries)})")
        
        texts = prepare_queries_for_extraction(self.queries, prompt_style="qa")
        query_ids = self.data_loader.get_query_ids(self.queries)
        
        target_layers = list(range(
            self.config['layers']['start'],
            self.config['layers']['end'] + 1
        ))
        print(f"\nExtracting activations from layers {target_layers}")
        
        extractor = ActivationExtractor(
            model=self.model,
            tokenizer=self.tokenizer,
            target_layers=target_layers,
            extraction_points=["hidden_states", "mlp"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.activations = extractor.extract_batch(
            texts=texts,
            query_ids=query_ids,
            batch_size=4,
            return_last_token=True,
            show_progress=True
        )
        
        activation_path = self.output_dir / "activations.pkl"
        save_activation_cache(self.activations, str(activation_path))
        print(f"Saved activations to {activation_path}")
        
        return self.activations
        
    def run_phase2_geometric_features(self):
        print("\n" + "="*60)
        print("PHASE 2: Geometric Feature Computation")
        print("="*60)
        
        if self.activations is None:
            activation_path = self.output_dir / "activations.pkl"
            if activation_path.exists():
                print("Loading activations from file...")
                self.activations = load_activation_cache(str(activation_path))
            else:
                raise ValueError("No activations available. Run phase 1 first.")
                
        activations_by_layer = {
            layer: get_layer_activations_as_matrix(self.activations, layer, "hidden")
            for layer in self.activations.hidden_states.keys()
        }
        
        print("\nComputing geometric features...")
        
        features_by_layer = compute_features_all_layers(
            activations_by_layer=activations_by_layer,
            labels=self.labels,
            query_ids=self.activations.query_ids,
            k_neighbors=self.config['feature_params']['k_neighbors'],
            pca_components=self.config['feature_params']['pca_components']
        )
        
        print("Aggregating features across layers...")
        aggregated = aggregate_features_across_layers(features_by_layer)
        
        self.geometric_features = pd.DataFrame(aggregated)
        self.geometric_features['dataset'] = self.labels
        
        query_ids = self.data_loader.get_query_ids(self.queries)
        self.geometric_features['query_id'] = query_ids
        
        answers = self.data_loader.get_answers(self.queries)
        self.geometric_features['answer'] = answers
        
        features_path = self.output_dir / "geometric_features.csv"
        self.geometric_features.to_csv(features_path, index=False)
        print(f"Saved features to {features_path}")
        
        print("\nFeature Statistics:")
        feature_cols = [c for c in self.geometric_features.columns 
                       if c not in ['query_id', 'dataset', 'answer']]
        print(self.geometric_features[feature_cols].describe())
        
        return self.geometric_features
        
    def run_phase3_attack(self):
        print("\n" + "="*60)
        print("PHASE 3: Activation Steering Attack")
        print("="*60)
        
        if self.model is None:
            self.model, self.tokenizer = setup_model(self.config)
            
        attack = AnonymizedActivationSteering(
            model=self.model,
            tokenizer=self.tokenizer,
            target_layer=self.config['attack'].get('target_layer', self.config['layers']['end']),
            steering_strength=self.config['attack']['steering_strength'],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print("\nCreating steering vectors...")
        sample_queries = self.queries[:20]
        original_questions = [q.question for q in sample_queries]
        
        anonymized_per_query = [
            attack.create_anonymized_questions(
                q, num_anonymizations=self.config['attack']['num_anonymizations']
            )
            for q in original_questions
        ]
        
        steering_vector = attack.compute_steering_vector(
            original_questions=original_questions,
            anonymized_questions_per_original=anonymized_per_query
        )
        
        print("\nRunning extraction attack...")
        questions = [q.question for q in self.queries]
        ground_truths = [q.answer for q in self.queries]
        query_ids = [q.query_id for q in self.queries]
        
        self.attack_results = []
        for i in tqdm(range(len(questions)), desc="Attacking"):
            result = attack.attack_single_query(
                question=questions[i],
                ground_truth=ground_truths[i],
                query_id=query_ids[i],
                steering_vector=steering_vector,
                num_samples=self.config['attack']['num_samples'],
                temperature=self.config['attack'].get('temperature', 2.0),
                max_new_tokens=self.config['attack'].get('max_new_tokens', 10)
            )
            self.attack_results.append(result)
            
        metrics = compute_attack_success_rate(self.attack_results)
        print(f"\nAttack Results:")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        print(f"  Average CAF: {metrics['average_caf']:.3f}")
        
        results_data = [
            {
                'query_id': r.query_id,
                'success': r.success,
                'caf': r.correct_answer_frequency,
                'extracted': r.extracted_answer,
                'ground_truth': r.ground_truth
            }
            for r in self.attack_results
        ]
        results_df = pd.DataFrame(results_data)
        results_path = self.output_dir / "attack_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Saved attack results to {results_path}")
        
        if self.geometric_features is not None:
            self.geometric_features = self.geometric_features.merge(
                results_df[['query_id', 'success', 'caf']],
                on='query_id',
                how='left'
            )
            
        return self.attack_results
        
    def run_phase4_prediction(self):
        print("\n" + "="*60)
        print("PHASE 4: Predictive Modeling")
        print("="*60)
        
        if self.geometric_features is None:
            features_path = self.output_dir / "geometric_features.csv"
            if features_path.exists():
                self.geometric_features = pd.read_csv(features_path)
            else:
                raise ValueError("No features available. Run phase 2 first.")
                
        if 'caf' not in self.geometric_features.columns:
            results_path = self.output_dir / "attack_results.csv"
            if results_path.exists():
                attack_df = pd.read_csv(results_path)
                self.geometric_features = self.geometric_features.merge(
                    attack_df[['query_id', 'success', 'caf']],
                    on='query_id',
                    how='left'
                )
            else:
                raise ValueError("No attack results available. Run phase 3 first.")
                
        feature_cols = [c for c in self.geometric_features.columns 
                       if any(x in c for x in ['density', 'separability', 'centrality', 
                                               'isolation', 'compactness', 'consistency'])]
        
        print(f"\nUsing {len(feature_cols)} features:")
        for col in feature_cols:
            print(f"  - {col}")
            
        print("\n--- Regression: Predicting CAF ---")
        reg_predictor = VulnerabilityPredictor(task_type="regression")
        X, y, feat_names = reg_predictor.prepare_data(
            self.geometric_features, target_col='caf', feature_cols=feature_cols
        )
        
        if len(X) < 10:
            raise ValueError(f"Not enough samples for prediction modeling: {len(X)}")
            
        reg_results = reg_predictor.train_all_models(X, y, feat_names)
        
        print("\n--- Classification: Predicting Success ---")
        clf_predictor = VulnerabilityPredictor(task_type="classification")
        X_clf, y_clf, _ = clf_predictor.prepare_data(
            self.geometric_features, target_col='success', feature_cols=feature_cols
        )
        clf_results = clf_predictor.train_all_models(X_clf, y_clf.astype(int), feat_names)
        
        print("\n--- Feature Importance Summary ---")
        importance_summary = create_feature_importance_summary(reg_results)
        if not importance_summary.empty:
            print(importance_summary.head(10))
            importance_summary.to_csv(
                self.output_dir / "feature_importance.csv", index=False
            )
            
        self.prediction_results = {
            'regression': reg_results,
            'classification': clf_results,
            'importance': importance_summary
        }
            
        return self.prediction_results
        
    def run_full_pipeline(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n{'='*60}")
        print(f"Started: {timestamp}")
        print(f"{'='*60}")
        
        self.run_phase1_data_and_activations()
        self.run_phase2_geometric_features()
        self.run_phase3_attack()
        self.run_phase4_prediction()
        
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Run experiment pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default="./outputs")
    parser.add_argument("--phase", type=int, default=0)
    args = parser.parse_args()
    
    config = load_config(args.config)
    pipeline = ExperimentPipeline(config, args.output)
    
    if args.phase == 0:
        pipeline.run_full_pipeline()
    elif args.phase == 1:
        pipeline.run_phase1_data_and_activations()
    elif args.phase == 2:
        pipeline.run_phase2_geometric_features()
    elif args.phase == 3:
        pipeline.run_phase3_attack()
    elif args.phase == 4:
        pipeline.run_phase4_prediction()
    else:
        raise ValueError(f"Unknown phase: {args.phase}")
        

if __name__ == "__main__":
    main()