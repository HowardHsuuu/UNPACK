"""
Multi-Attack Analysis
Separates analysis by attack type:
- Embedding (Oracle): Does knowledge still exist?
- Steering (Realistic): Can it be extracted via activation steering?
- Prompt (Realistic): Can it be extracted via prompt manipulation?
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

EXPERIMENTS = {
    'HP': './outputs_hp_full',
    'TOFU_grad_diff': './outputs_tofu_1_grad_diff',
    'TOFU_grad_ascent': './outputs_tofu_2_grad_ascent',
    'TOFU_KL': './outputs_tofu_3_KL',
    'TOFU_idk': './outputs_tofu_4_idk',
}

ALL_FEATURES = [
    'local_density_mean', 'local_density_std', 'local_density_max', 'local_density_min',
    'separability_mean', 'separability_std', 'separability_max', 'separability_min',
    'centrality_mean', 'centrality_std', 'centrality_max', 'centrality_min',
    'isolation_mean', 'isolation_std', 'isolation_max', 'isolation_min',
    'cluster_compactness_mean', 'cluster_compactness_std', 'cluster_compactness_max', 'cluster_compactness_min',
    'cross_layer_consistency_mean', 'cross_layer_consistency_std', 'cross_layer_consistency_max', 'cross_layer_consistency_min',
]

ATTACK_TYPES = {
    'embedding': ('embedding_attack_results.csv', 'Oracle (Knowledge Exists?)'),
    'steering': ('activation_steering_results.csv', 'Realistic (Steering)'),
    'prompt': ('prompt_attack_results.csv', 'Realistic (Prompt)'),
}


def load_experiment(name, path):
    path = Path(path)
    if not path.exists():
        return None
    
    data = {'name': name}
    
    # Load features
    for key, filename in [('base_features', 'base_geometric_features.csv'),
                          ('unlearned_features', 'unlearned_geometric_features.csv'),
                          ('retention', 'direct_retention.csv')]:
        filepath = path / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
    
    # Load all attack results
    data['attacks'] = {}
    for attack_name, (filename, description) in ATTACK_TYPES.items():
        filepath = path / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            # Normalize column name
            if 'caf' not in df.columns and 'best_caf' in df.columns:
                df['caf'] = df['best_caf']
            data['attacks'][attack_name] = df
    
    if 'base_features' not in data or 'retention' not in data:
        return None
    
    return data


def prepare_outcomes(data):
    """Prepare outcomes with all attack types"""
    retention = data['retention'].copy()
    retention['unlearning_success'] = 1 - retention['retention']
    
    outcomes = retention[['query_id', 'unlearning_success']].copy()
    
    for attack_name, attack_df in data['attacks'].items():
        if 'caf' in attack_df.columns:
            outcomes = outcomes.merge(
                attack_df[['query_id', 'caf']].rename(columns={'caf': f'{attack_name}_caf'}),
                on='query_id', how='left'
            )
    
    return outcomes


def analyze_correlations_by_attack(experiments):
    """Analyze correlations for each attack type separately"""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS BY ATTACK TYPE")
    print("="*80)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_outcomes(data)
        available_features = [f for f in ALL_FEATURES if f in data['base_features'].columns]
        
        # Get available attack types
        attack_cols = [c for c in outcomes.columns if c.endswith('_caf')]
        targets = [('unlearning_success', 'Unlearning')] + [(c, c.replace('_caf', '').title()) for c in attack_cols]
        
        for geom_name, feat_df in [('base', data['base_features']), 
                                    ('unlearned', data.get('unlearned_features'))]:
            if feat_df is None:
                continue
            
            merged = feat_df.merge(outcomes, on='query_id')
            
            for target_col, target_name in targets:
                if target_col not in merged.columns:
                    continue
                    
                for feat in available_features:
                    if feat not in merged.columns:
                        continue
                    
                    x = merged[feat].values
                    y = merged[target_col].values
                    
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x_clean, y_clean = x[mask], y[mask]
                    
                    if len(x_clean) < 10:
                        continue
                    
                    r, p = pearsonr(x_clean, y_clean)
                    
                    all_results.append({
                        'experiment': exp_name,
                        'geometry': geom_name,
                        'target': target_name,
                        'feature': feat,
                        'pearson_r': r,
                        'p_value': p,
                        'significant': p < 0.05,
                        'direction': '+' if r > 0 else '-',
                        'n': len(x_clean)
                    })
    
    df = pd.DataFrame(all_results)
    
    # Print summary by attack type
    for target in df['target'].unique():
        subset = df[df['target'] == target]
        sig = subset[subset['significant']]
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")
        print(f"Total correlations: {len(subset)}, Significant: {len(sig)} ({len(sig)/len(subset)*100:.1f}%)")
        
        if len(sig) > 0:
            print(f"\nTop significant correlations:")
            print(f"{'Experiment':<18} {'Geom':<10} {'Feature':<30} {'r':>8} {'Dir'}")
            print("-" * 75)
            for _, row in sig.nlargest(10, 'pearson_r', keep='all').iterrows():
                print(f"{row['experiment']:<18} {row['geometry']:<10} {row['feature']:<30} {row['pearson_r']:>8.3f} {row['direction']}")
    
    return df


def analyze_prediction_by_attack(experiments):
    """Prediction analysis for each attack type"""
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS BY ATTACK TYPE")
    print("="*80)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_outcomes(data)
        available_features = [f for f in ALL_FEATURES if f in data['base_features'].columns]
        
        attack_cols = [c for c in outcomes.columns if c.endswith('_caf')]
        targets = [('unlearning_success', 'Unlearning')] + [(c, c.replace('_caf', '').title()) for c in attack_cols]
        
        for geom_name, feat_df in [('base', data['base_features']), 
                                    ('unlearned', data.get('unlearned_features'))]:
            if feat_df is None:
                continue
            
            merged = feat_df.merge(outcomes, on='query_id')
            
            for target_col, target_name in targets:
                if target_col not in merged.columns:
                    continue
                
                X = merged[available_features].values
                y = merged[target_col].values
                
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X, y = X[mask], y[mask]
                
                if len(X) < 20:
                    continue
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Linear
                lr = LinearRegression()
                cv_lr = cross_val_score(lr, X_scaled, y, cv=5, scoring='r2')
                
                # Decision Tree
                dt = DecisionTreeRegressor(max_depth=4, random_state=42)
                cv_dt = cross_val_score(dt, X_scaled, y, cv=5, scoring='r2')
                
                # Random Forest
                rf = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
                cv_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')
                
                all_results.append({
                    'experiment': exp_name,
                    'geometry': geom_name,
                    'target': target_name,
                    'linear_r2': cv_lr.mean(),
                    'tree_r2': cv_dt.mean(),
                    'rf_r2': cv_rf.mean(),
                    'best_nonlinear': max(cv_dt.mean(), cv_rf.mean()),
                    'nonlinear_advantage': max(cv_dt.mean(), cv_rf.mean()) - cv_lr.mean(),
                    'n': len(X)
                })
    
    df = pd.DataFrame(all_results)
    
    # Print summary by attack type
    for target in df['target'].unique():
        subset = df[df['target'] == target]
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")
        print(f"{'Experiment':<18} {'Geom':<10} {'Linear':>10} {'Tree':>10} {'RF':>10} {'NL Adv':>10}")
        print("-" * 70)
        for _, row in subset.iterrows():
            print(f"{row['experiment']:<18} {row['geometry']:<10} {row['linear_r2']:>10.3f} "
                  f"{row['tree_r2']:>10.3f} {row['rf_r2']:>10.3f} {row['nonlinear_advantage']:>+10.3f}")
    
    return df


def analyze_thresholds_by_attack(experiments):
    """Find threshold rules for each attack type"""
    print("\n" + "="*80)
    print("THRESHOLD RULES BY ATTACK TYPE")
    print("="*80)
    
    all_rules = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_outcomes(data)
        available_features = [f for f in ALL_FEATURES if f in data['base_features'].columns]
        
        attack_cols = [c for c in outcomes.columns if c.endswith('_caf')]
        targets = [('unlearning_success', 'Unlearning')] + [(c, c.replace('_caf', '').title()) for c in attack_cols]
        
        for geom_name, feat_df in [('base', data['base_features']), 
                                    ('unlearned', data.get('unlearned_features'))]:
            if feat_df is None:
                continue
            
            merged = feat_df.merge(outcomes, on='query_id')
            
            for target_col, target_name in targets:
                if target_col not in merged.columns:
                    continue
                
                X = merged[available_features].values
                y = merged[target_col].values
                
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X, y = X[mask], y[mask]
                
                if len(X) < 20:
                    continue
                
                # Train decision tree
                dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=42)
                dt.fit(X, y)
                
                cv_scores = cross_val_score(dt, X, y, cv=5, scoring='r2')
                
                if cv_scores.mean() > -0.3:  # Only show if somewhat predictive
                    print(f"\n{exp_name} / {geom_name} -> {target_name} (CV R2: {cv_scores.mean():.3f})")
                    print("-" * 50)
                    rules = export_text(dt, feature_names=available_features, max_depth=3)
                    print(rules)
                    
                    all_rules.append({
                        'experiment': exp_name,
                        'geometry': geom_name,
                        'target': target_name,
                        'cv_r2': cv_scores.mean(),
                        'rules': rules
                    })
    
    return pd.DataFrame(all_rules)


def analyze_cross_attack_consistency(corr_df):
    """Find features that predict multiple attack types consistently"""
    print("\n" + "="*80)
    print("CROSS-ATTACK CONSISTENCY")
    print("Which features predict multiple attack types?")
    print("="*80)
    
    sig = corr_df[corr_df['significant']].copy()
    
    if len(sig) == 0:
        print("No significant correlations found")
        return pd.DataFrame()
    
    # Group by feature and geometry
    consistency = sig.groupby(['feature', 'geometry']).agg({
        'target': lambda x: list(x.unique()),
        'pearson_r': 'mean',
        'direction': lambda x: x.mode()[0] if len(x) > 0 else '?',
        'experiment': lambda x: len(x.unique())
    }).reset_index()
    
    consistency['n_targets'] = consistency['target'].apply(len)
    consistency = consistency.sort_values('n_targets', ascending=False)
    
    print(f"\nFeatures significant for multiple targets:")
    print(f"{'Feature':<30} {'Geom':<10} {'#Targets':>10} {'Mean r':>10} {'Dir':<5} {'Targets'}")
    print("-" * 90)
    
    for _, row in consistency[consistency['n_targets'] >= 2].iterrows():
        targets_str = ', '.join(row['target'])[:30]
        print(f"{row['feature']:<30} {row['geometry']:<10} {row['n_targets']:>10} "
              f"{row['pearson_r']:>10.3f} {row['direction']:<5} {targets_str}")
    
    return consistency


def compute_summary(experiments):
    """Compute summary statistics"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    rows = []
    for exp_name, data in experiments.items():
        row = {'experiment': exp_name}
        row['n_queries'] = len(data['base_features'])
        
        if 'retention' in data:
            row['retention'] = data['retention']['retention'].mean()
            row['unlearning'] = 1 - row['retention']
        
        for attack_name, attack_df in data['attacks'].items():
            if 'caf' in attack_df.columns:
                row[f'{attack_name}_caf'] = attack_df['caf'].mean()
                row[f'{attack_name}_success'] = (attack_df['caf'] > 0).mean()
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    print("\n" + df.to_string(index=False))
    
    return df


def main():
    print("\n" + "#"*80)
    print("# MULTI-ATTACK ANALYSIS")
    print("# Analyzing each attack type separately")
    print("#"*80)
    
    # Load experiments
    print("\n" + "="*80)
    print("LOADING EXPERIMENTS")
    print("="*80)
    
    experiments = {}
    for name, path in EXPERIMENTS.items():
        data = load_experiment(name, path)
        if data:
            attacks_loaded = list(data['attacks'].keys())
            print(f"  [OK] {name}: {attacks_loaded}")
            experiments[name] = data
        else:
            print(f"  [--] {name}: not found")
    
    if len(experiments) == 0:
        print("No experiments found!")
        return
    
    # Run analyses
    results = {}
    
    results['summary'] = compute_summary(experiments)
    results['correlations'] = analyze_correlations_by_attack(experiments)
    results['predictions'] = analyze_prediction_by_attack(experiments)
    results['consistency'] = analyze_cross_attack_consistency(results['correlations'])
    results['thresholds'] = analyze_thresholds_by_attack(experiments)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_dir = Path('./multi_attack_analysis')
    output_dir.mkdir(exist_ok=True)
    
    for name, df in results.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            path = output_dir / f'{name}.csv'
            df.to_csv(path, index=False)
            print(f"  [OK] {path}")
    
    # Final summary
    print("\n" + "#"*80)
    print("# KEY INSIGHTS BY ATTACK TYPE")
    print("#"*80)
    
    corr = results['correlations']
    pred = results['predictions']
    
    for target in corr['target'].unique():
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")
        
        # Best correlation
        target_corr = corr[(corr['target'] == target) & (corr['significant'])]
        if len(target_corr) > 0:
            best = target_corr.loc[target_corr['pearson_r'].abs().idxmax()]
            print(f"Best correlation: {best['feature']} (r={best['pearson_r']:.3f}, {best['experiment']})")
        
        # Best prediction
        target_pred = pred[pred['target'] == target]
        if len(target_pred) > 0:
            best_pred = target_pred.loc[target_pred['best_nonlinear'].idxmax()]
            print(f"Best prediction:  {best_pred['experiment']}/{best_pred['geometry']} "
                  f"(RF R2={best_pred['rf_r2']:.3f}, NL advantage={best_pred['nonlinear_advantage']:+.3f})")
    
    print("\n" + "#"*80)
    print("# ANALYSIS COMPLETE")
    print(f"# Results saved to: {output_dir}")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
