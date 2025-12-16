"""
Comprehensive Cross-Experiment Analysis
Excludes Embedding Attack (Oracle) - focuses on realistic attacks only

Targets:
- Unlearning Success (1 - retention)
- Steering CAF (realistic attack)
- Prompt CAF (realistic attack)

Analyses:
1. Experiment Summary
2. Correlation Analysis (Pearson + Spearman)
3. Single Feature Prediction
4. Multi-Feature Prediction (Linear vs NonLinear)
5. Decision Tree Rules (Interpretable thresholds)
6. Cross-Experiment Consistency
7. Cross-Attack Consistency
8. Base vs Unlearned Geometry Comparison
9. Non-Monotonic Relationship Detection
10. Permutation Importance
11. Feature Group Analysis
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

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

FEATURE_GROUPS = {
    'density': ['local_density_mean', 'local_density_std', 'local_density_max', 'local_density_min'],
    'separability': ['separability_mean', 'separability_std', 'separability_max', 'separability_min'],
    'centrality': ['centrality_mean', 'centrality_std', 'centrality_max', 'centrality_min'],
    'isolation': ['isolation_mean', 'isolation_std', 'isolation_max', 'isolation_min'],
    'compactness': ['cluster_compactness_mean', 'cluster_compactness_std', 'cluster_compactness_max', 'cluster_compactness_min'],
    'consistency': ['cross_layer_consistency_mean', 'cross_layer_consistency_std', 'cross_layer_consistency_max', 'cross_layer_consistency_min'],
}

# Only realistic attacks (exclude embedding/oracle)
ATTACK_FILES = {
    'steering': 'activation_steering_results.csv',
    'prompt': 'prompt_attack_results.csv',
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_experiment(name, path):
    """Load all data for one experiment"""
    path = Path(path)
    if not path.exists():
        return None
    
    data = {'name': name}
    
    # Core files
    for key, filename in [
        ('base_features', 'base_geometric_features.csv'),
        ('unlearned_features', 'unlearned_geometric_features.csv'),
        ('retention', 'direct_retention.csv'),
        ('base_accuracy', 'base_accuracy.csv'),
    ]:
        filepath = path / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
    
    # Attack results (realistic only)
    data['attacks'] = {}
    for attack_name, filename in ATTACK_FILES.items():
        filepath = path / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            if 'caf' not in df.columns and 'best_caf' in df.columns:
                df['caf'] = df['best_caf']
            data['attacks'][attack_name] = df
    
    if 'base_features' not in data or 'retention' not in data:
        return None
    
    return data


def load_all_experiments():
    """Load all available experiments"""
    experiments = {}
    for name, path in EXPERIMENTS.items():
        data = load_experiment(name, path)
        if data:
            experiments[name] = data
    return experiments


def prepare_outcomes(data):
    """Prepare outcome variables"""
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


def get_targets(outcomes):
    """Get available target columns"""
    targets = [('unlearning_success', 'Unlearning')]
    for col in outcomes.columns:
        if col.endswith('_caf'):
            name = col.replace('_caf', '').title()
            targets.append((col, name))
    return targets


# =============================================================================
# ANALYSIS 1: EXPERIMENT SUMMARY
# =============================================================================

def analyze_summary(experiments):
    """Compute summary statistics for all experiments"""
    print("\n" + "="*80)
    print("ANALYSIS 1: EXPERIMENT SUMMARY")
    print("="*80)
    
    rows = []
    for exp_name, data in experiments.items():
        row = {'experiment': exp_name}
        row['n_queries'] = len(data['base_features'])
        
        if 'base_accuracy' in data:
            row['base_caf'] = data['base_accuracy']['base_caf'].mean()
        
        if 'retention' in data:
            row['retention'] = data['retention']['retention'].mean()
            row['unlearning_eff'] = 1 - row['retention']
        
        for attack_name, attack_df in data['attacks'].items():
            if 'caf' in attack_df.columns:
                row[f'{attack_name}_caf'] = attack_df['caf'].mean()
                row[f'{attack_name}_success'] = (attack_df['caf'] > 0).mean()
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    
    return df


# =============================================================================
# ANALYSIS 2: CORRELATION ANALYSIS
# =============================================================================

def analyze_correlations(experiments):
    """Compute correlations between features and targets"""
    print("\n" + "="*80)
    print("ANALYSIS 2: CORRELATION ANALYSIS")
    print("="*80)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_outcomes(data)
        targets = get_targets(outcomes)
        available_features = [f for f in ALL_FEATURES if f in data['base_features'].columns]
        
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
                    
                    r_pearson, p_pearson = pearsonr(x_clean, y_clean)
                    r_spearman, p_spearman = spearmanr(x_clean, y_clean)
                    
                    all_results.append({
                        'experiment': exp_name,
                        'geometry': geom_name,
                        'target': target_name,
                        'feature': feat,
                        'pearson_r': r_pearson,
                        'pearson_p': p_pearson,
                        'spearman_r': r_spearman,
                        'spearman_p': p_spearman,
                        'significant': p_pearson < 0.05,
                        'direction': '+' if r_pearson > 0 else '-',
                        'n': len(x_clean)
                    })
    
    df = pd.DataFrame(all_results)
    
    # Print summary by target
    for target in df['target'].unique():
        subset = df[df['target'] == target]
        sig = subset[subset['significant']]
        print(f"\n--- {target} ---")
        print(f"Total: {len(subset)}, Significant: {len(sig)} ({len(sig)/len(subset)*100:.1f}%)")
        
        if len(sig) > 0:
            top = sig.reindex(sig['pearson_r'].abs().sort_values(ascending=False).index).head(5)
            for _, row in top.iterrows():
                print(f"  {row['direction']} {row['feature']}: r={row['pearson_r']:.3f} ({row['experiment']}/{row['geometry']})")
    
    return df


# =============================================================================
# ANALYSIS 3: SINGLE FEATURE PREDICTION
# =============================================================================

def analyze_single_feature(experiments):
    """Test each feature individually as predictor"""
    print("\n" + "="*80)
    print("ANALYSIS 3: SINGLE FEATURE PREDICTION")
    print("="*80)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_outcomes(data)
        targets = get_targets(outcomes)
        available_features = [f for f in ALL_FEATURES if f in data['base_features'].columns]
        
        for geom_name, feat_df in [('base', data['base_features']), 
                                    ('unlearned', data.get('unlearned_features'))]:
            if feat_df is None:
                continue
            
            merged = feat_df.merge(outcomes, on='query_id')
            
            for target_col, target_name in targets:
                if target_col not in merged.columns:
                    continue
                
                for feat in available_features:
                    X = merged[feat].values.reshape(-1, 1)
                    y = merged[target_col].values
                    
                    mask = ~(np.isnan(X.flatten()) | np.isnan(y))
                    X, y = X[mask], y[mask]
                    
                    if len(X) < 15:
                        continue
                    
                    r, p = pearsonr(X.flatten(), y)
                    
                    model = LinearRegression()
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                    
                    all_results.append({
                        'experiment': exp_name,
                        'geometry': geom_name,
                        'target': target_name,
                        'feature': feat,
                        'pearson_r': r,
                        'p_value': p,
                        'significant': p < 0.05,
                        'cv_r2_mean': cv_scores.mean(),
                        'cv_r2_std': cv_scores.std(),
                        'direction': '+' if r > 0 else '-'
                    })
    
    df = pd.DataFrame(all_results)
    
    # Best features per target
    print("\nBest single-feature predictors (by |r|):")
    for target in df['target'].unique():
        subset = df[df['target'] == target]
        best = subset.loc[subset['pearson_r'].abs().idxmax()]
        print(f"  {target}: {best['feature']} (r={best['pearson_r']:.3f}, {best['experiment']})")
    
    return df


# =============================================================================
# ANALYSIS 4: MULTI-FEATURE PREDICTION (LINEAR vs NONLINEAR)
# =============================================================================

def analyze_multi_feature(experiments):
    """Compare linear vs nonlinear models with all features"""
    print("\n" + "="*80)
    print("ANALYSIS 4: MULTI-FEATURE PREDICTION (LINEAR vs NONLINEAR)")
    print("="*80)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_outcomes(data)
        targets = get_targets(outcomes)
        available_features = [f for f in ALL_FEATURES if f in data['base_features'].columns]
        
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
                
                models = {
                    'Linear': LinearRegression(),
                    'Ridge': Ridge(alpha=1.0),
                    'DecisionTree': DecisionTreeRegressor(max_depth=4, random_state=42),
                    'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42),
                    'GradientBoosting': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
                }
                
                for model_name, model in models.items():
                    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                    
                    all_results.append({
                        'experiment': exp_name,
                        'geometry': geom_name,
                        'target': target_name,
                        'model': model_name,
                        'cv_r2_mean': cv_scores.mean(),
                        'cv_r2_std': cv_scores.std(),
                        'n_features': len(available_features),
                        'n_samples': len(X)
                    })
    
    df = pd.DataFrame(all_results)
    
    # Summary table
    print("\nLinear vs NonLinear (best of Tree/RF/GB):")
    print(f"{'Experiment':<18} {'Geom':<10} {'Target':<12} {'Linear':>10} {'NonLinear':>10} {'Advantage':>10}")
    print("-" * 75)
    
    for (exp, geom, target), group in df.groupby(['experiment', 'geometry', 'target']):
        linear = group[group['model'] == 'Linear']['cv_r2_mean'].values[0]
        nonlinear = group[group['model'].isin(['DecisionTree', 'RandomForest', 'GradientBoosting'])]['cv_r2_mean'].max()
        advantage = nonlinear - linear
        marker = "***" if advantage > 0.1 else ""
        print(f"{exp:<18} {geom:<10} {target:<12} {linear:>10.3f} {nonlinear:>10.3f} {advantage:>+10.3f} {marker}")
    
    return df


# =============================================================================
# ANALYSIS 5: DECISION TREE RULES
# =============================================================================

def analyze_tree_rules(experiments):
    """Extract interpretable rules from decision trees"""
    print("\n" + "="*80)
    print("ANALYSIS 5: DECISION TREE RULES (Interpretable Thresholds)")
    print("="*80)
    
    all_rules = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_outcomes(data)
        targets = get_targets(outcomes)
        available_features = [f for f in ALL_FEATURES if f in data['base_features'].columns]
        
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
                
                tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=42)
                tree.fit(X, y)
                
                cv_scores = cross_val_score(tree, X, y, cv=5, scoring='r2')
                
                if cv_scores.mean() > -0.2:  # Only show if somewhat predictive
                    rules = export_text(tree, feature_names=available_features, max_depth=3)
                    
                    print(f"\n{exp_name} / {geom_name} -> {target_name} (CV R2: {cv_scores.mean():.3f})")
                    print("-" * 50)
                    print(rules)
                    
                    all_rules.append({
                        'experiment': exp_name,
                        'geometry': geom_name,
                        'target': target_name,
                        'cv_r2': cv_scores.mean(),
                        'rules': rules
                    })
    
    return pd.DataFrame(all_rules)


# =============================================================================
# ANALYSIS 6: CROSS-EXPERIMENT CONSISTENCY
# =============================================================================

def analyze_cross_experiment(corr_df):
    """Find features significant across multiple experiments"""
    print("\n" + "="*80)
    print("ANALYSIS 6: CROSS-EXPERIMENT CONSISTENCY")
    print("="*80)
    
    sig = corr_df[corr_df['significant']].copy()
    
    if len(sig) == 0:
        print("No significant correlations found")
        return pd.DataFrame()
    
    consistency = sig.groupby(['feature', 'geometry', 'target']).agg({
        'pearson_r': ['mean', 'std', 'count'],
        'direction': lambda x: x.mode()[0] if len(x) > 0 else '?',
        'experiment': lambda x: list(x.unique())
    }).reset_index()
    
    consistency.columns = ['feature', 'geometry', 'target', 'mean_r', 'std_r', 
                          'n_experiments', 'direction', 'experiments']
    
    consistent = consistency[consistency['n_experiments'] >= 2].sort_values(
        'n_experiments', ascending=False
    )
    
    print(f"\nFeatures significant in 2+ experiments:")
    print(f"{'Feature':<30} {'Geom':<10} {'Target':<12} {'n':>3} {'Mean r':>8} {'Dir'}")
    print("-" * 75)
    
    for _, row in consistent.iterrows():
        print(f"{row['feature']:<30} {row['geometry']:<10} {row['target']:<12} "
              f"{row['n_experiments']:>3} {row['mean_r']:>8.3f} {row['direction']}")
    
    return consistent


# =============================================================================
# ANALYSIS 7: CROSS-ATTACK CONSISTENCY
# =============================================================================

def analyze_cross_attack(corr_df):
    """Find features that predict multiple attack types"""
    print("\n" + "="*80)
    print("ANALYSIS 7: CROSS-ATTACK CONSISTENCY")
    print("="*80)
    
    sig = corr_df[corr_df['significant']].copy()
    
    if len(sig) == 0:
        print("No significant correlations found")
        return pd.DataFrame()
    
    consistency = sig.groupby(['feature', 'geometry']).agg({
        'target': lambda x: list(x.unique()),
        'pearson_r': 'mean',
        'direction': lambda x: x.mode()[0] if len(x) > 0 else '?'
    }).reset_index()
    
    consistency['n_targets'] = consistency['target'].apply(len)
    consistency = consistency.sort_values('n_targets', ascending=False)
    
    multi_target = consistency[consistency['n_targets'] >= 2]
    
    print(f"\nFeatures predicting multiple targets:")
    print(f"{'Feature':<30} {'Geom':<10} {'#Targets':>10} {'Mean r':>10} {'Targets'}")
    print("-" * 85)
    
    for _, row in multi_target.head(15).iterrows():
        targets_str = ', '.join(row['target'])[:30]
        print(f"{row['feature']:<30} {row['geometry']:<10} {row['n_targets']:>10} "
              f"{row['pearson_r']:>10.3f} {targets_str}")
    
    return consistency


# =============================================================================
# ANALYSIS 8: BASE vs UNLEARNED GEOMETRY
# =============================================================================

def analyze_geometry_change(experiments):
    """Compare geometry before and after unlearning"""
    print("\n" + "="*80)
    print("ANALYSIS 8: BASE vs UNLEARNED GEOMETRY COMPARISON")
    print("="*80)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        if 'unlearned_features' not in data:
            continue
        
        base = data['base_features']
        unlearned = data['unlearned_features']
        
        merged = base.merge(unlearned, on='query_id', suffixes=('_base', '_unlearned'))
        
        available = [f for f in ALL_FEATURES if f in base.columns]
        
        for feat in available:
            base_col = f"{feat}_base"
            unlearned_col = f"{feat}_unlearned"
            
            if base_col not in merged.columns:
                continue
            
            base_vals = merged[base_col].dropna()
            unlearned_vals = merged[unlearned_col].dropna()
            
            if len(base_vals) < 10:
                continue
            
            t_stat, p_val = ttest_ind(base_vals, unlearned_vals)
            
            pooled_std = np.sqrt((base_vals.std()**2 + unlearned_vals.std()**2) / 2)
            cohens_d = (unlearned_vals.mean() - base_vals.mean()) / pooled_std if pooled_std > 0 else 0
            
            all_results.append({
                'experiment': exp_name,
                'feature': feat,
                'base_mean': base_vals.mean(),
                'unlearned_mean': unlearned_vals.mean(),
                'change_pct': (unlearned_vals.mean() - base_vals.mean()) / base_vals.mean() * 100 if base_vals.mean() != 0 else 0,
                'cohens_d': cohens_d,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
    
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("No comparison data available")
        return df
    
    # Show significant changes
    sig = df[(df['significant']) & (df['cohens_d'].abs() > 0.3)].sort_values('cohens_d', key=abs, ascending=False)
    
    print(f"\nSignificant geometry changes (|Cohen's d| > 0.3):")
    print(f"{'Experiment':<18} {'Feature':<30} {'Change%':>10} {'Cohen d':>10}")
    print("-" * 75)
    
    for _, row in sig.head(15).iterrows():
        print(f"{row['experiment']:<18} {row['feature']:<30} "
              f"{row['change_pct']:>+10.1f}% {row['cohens_d']:>+10.3f}")
    
    return df


# =============================================================================
# ANALYSIS 9: NON-MONOTONIC RELATIONSHIPS
# =============================================================================

def analyze_nonmonotonic(experiments):
    """Detect U-shaped or inverted-U relationships"""
    print("\n" + "="*80)
    print("ANALYSIS 9: NON-MONOTONIC RELATIONSHIPS")
    print("="*80)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_outcomes(data)
        targets = get_targets(outcomes)
        available_features = [f for f in ALL_FEATURES if f in data['base_features'].columns]
        
        for geom_name, feat_df in [('base', data['base_features']), 
                                    ('unlearned', data.get('unlearned_features'))]:
            if feat_df is None:
                continue
            
            merged = feat_df.merge(outcomes, on='query_id')
            
            for target_col, target_name in targets:
                if target_col not in merged.columns:
                    continue
                
                for feat in available_features:
                    x = merged[feat].values
                    y = merged[target_col].values
                    
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[mask], y[mask]
                    
                    if len(x) < 20:
                        continue
                    
                    try:
                        bins = pd.qcut(x, q=4, duplicates='drop')
                        bin_means = pd.Series(y).groupby(bins).mean().values
                        
                        if len(bin_means) < 3:
                            continue
                        
                        # Check monotonicity
                        mono_inc = all(bin_means[i] <= bin_means[i+1] for i in range(len(bin_means)-1))
                        mono_dec = all(bin_means[i] >= bin_means[i+1] for i in range(len(bin_means)-1))
                        
                        if not mono_inc and not mono_dec:
                            mid = len(bin_means) // 2
                            if bin_means[0] > bin_means[mid] < bin_means[-1]:
                                shape = "U-shaped"
                            elif bin_means[0] < bin_means[mid] > bin_means[-1]:
                                shape = "Inverted-U"
                            else:
                                shape = "Complex"
                            
                            range_y = bin_means.max() - bin_means.min()
                            
                            if range_y > 0.05:
                                all_results.append({
                                    'experiment': exp_name,
                                    'geometry': geom_name,
                                    'target': target_name,
                                    'feature': feat,
                                    'shape': shape,
                                    'range': range_y,
                                    'bin_means': list(bin_means.round(3))
                                })
                    except:
                        continue
    
    df = pd.DataFrame(all_results)
    
    if len(df) > 0:
        print(f"\nNon-monotonic relationships detected: {len(df)}")
        shapes = df['shape'].value_counts()
        for shape, count in shapes.items():
            print(f"  {shape}: {count}")
        
        print(f"\nTop non-monotonic (by range):")
        for _, row in df.nlargest(10, 'range').iterrows():
            print(f"  {row['experiment']}/{row['geometry']} -> {row['target']}: {row['feature']} ({row['shape']})")
            print(f"    Bin means: {row['bin_means']}")
    
    return df


# =============================================================================
# ANALYSIS 10: PERMUTATION IMPORTANCE
# =============================================================================

def analyze_permutation_importance(experiments):
    """Compute permutation importance with RandomForest"""
    print("\n" + "="*80)
    print("ANALYSIS 10: PERMUTATION IMPORTANCE (RandomForest)")
    print("="*80)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_outcomes(data)
        targets = get_targets(outcomes)
        available_features = [f for f in ALL_FEATURES if f in data['base_features'].columns]
        
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
                
                if len(X) < 30:
                    continue
                
                rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                rf.fit(X, y)
                
                cv_r2 = cross_val_score(rf, X, y, cv=5, scoring='r2').mean()
                
                if cv_r2 < -0.5:
                    continue
                
                perm_imp = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
                
                for i, feat in enumerate(available_features):
                    imp = perm_imp.importances_mean[i]
                    if imp > 0.001:
                        all_results.append({
                            'experiment': exp_name,
                            'geometry': geom_name,
                            'target': target_name,
                            'feature': feat,
                            'importance': imp,
                            'importance_std': perm_imp.importances_std[i],
                            'cv_r2': cv_r2
                        })
    
    df = pd.DataFrame(all_results)
    
    if len(df) > 0:
        print(f"\nTop important features (aggregated):")
        agg = df.groupby('feature')['importance'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        for feat, row in agg.head(10).iterrows():
            print(f"  {feat}: mean_imp={row['mean']:.4f} (n={int(row['count'])})")
    
    return df


# =============================================================================
# ANALYSIS 11: FEATURE GROUP ANALYSIS
# =============================================================================

def analyze_feature_groups(experiments):
    """Test feature groups as predictors"""
    print("\n" + "="*80)
    print("ANALYSIS 11: FEATURE GROUP ANALYSIS")
    print("="*80)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_outcomes(data)
        targets = get_targets(outcomes)
        
        for geom_name, feat_df in [('base', data['base_features']), 
                                    ('unlearned', data.get('unlearned_features'))]:
            if feat_df is None:
                continue
            
            merged = feat_df.merge(outcomes, on='query_id')
            
            for target_col, target_name in targets:
                if target_col not in merged.columns:
                    continue
                
                for group_name, group_features in FEATURE_GROUPS.items():
                    available = [f for f in group_features if f in merged.columns]
                    if len(available) == 0:
                        continue
                    
                    X = merged[available].values
                    y = merged[target_col].values
                    
                    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                    X, y = X[mask], y[mask]
                    
                    if len(X) < 15:
                        continue
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    lr = LinearRegression()
                    cv_lr = cross_val_score(lr, X_scaled, y, cv=5, scoring='r2')
                    
                    rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
                    cv_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')
                    
                    all_results.append({
                        'experiment': exp_name,
                        'geometry': geom_name,
                        'target': target_name,
                        'group': group_name,
                        'linear_r2': cv_lr.mean(),
                        'rf_r2': cv_rf.mean(),
                        'n_features': len(available)
                    })
    
    df = pd.DataFrame(all_results)
    
    # Pivot table
    print("\nFeature Group Performance (RF R2):")
    if len(df) > 0:
        pivot = df.pivot_table(values='rf_r2', index=['experiment', 'geometry'], 
                               columns=['target', 'group'], aggfunc='mean')
        print(pivot.round(3).to_string())
    
    return df


# =============================================================================
# MAIN REPORT
# =============================================================================

def generate_key_findings(results):
    """Generate key findings summary"""
    print("\n" + "#"*80)
    print("# KEY FINDINGS SUMMARY")
    print("#"*80)
    
    # 1. Best predictors per target
    print("\n1. BEST PREDICTORS BY TARGET (Correlation)")
    corr = results['correlations']
    for target in corr['target'].unique():
        sig = corr[(corr['target'] == target) & (corr['significant'])]
        if len(sig) > 0:
            best = sig.loc[sig['pearson_r'].abs().idxmax()]
            print(f"   {target}: {best['feature']} (r={best['pearson_r']:.3f}, {best['experiment']})")
    
    # 2. Nonlinear advantage
    print("\n2. NONLINEAR ADVANTAGE (Linear vs Tree/RF)")
    pred = results['predictions']
    for target in pred['target'].unique():
        subset = pred[pred['target'] == target]
        linear_avg = subset[subset['model'] == 'Linear']['cv_r2_mean'].mean()
        nonlinear_avg = subset[subset['model'].isin(['DecisionTree', 'RandomForest'])]['cv_r2_mean'].max()
        print(f"   {target}: Linear avg={linear_avg:.3f}, NonLinear best={nonlinear_avg:.3f}, Δ={nonlinear_avg-linear_avg:+.3f}")
    
    # 3. Cross-experiment consistent features
    print("\n3. CROSS-EXPERIMENT CONSISTENT FEATURES")
    consistent = results['cross_experiment']
    if len(consistent) > 0:
        for _, row in consistent.head(5).iterrows():
            print(f"   {row['feature']} ({row['geometry']} -> {row['target']}): n={row['n_experiments']}, r={row['mean_r']:.3f}")
    
    # 4. Geometry changes
    print("\n4. SIGNIFICANT GEOMETRY CHANGES AFTER UNLEARNING")
    geom = results['geometry_change']
    sig_geom = geom[(geom['significant']) & (geom['cohens_d'].abs() > 0.5)]
    if len(sig_geom) > 0:
        for _, row in sig_geom.head(5).iterrows():
            print(f"   {row['experiment']}/{row['feature']}: Cohen's d={row['cohens_d']:+.2f}")
    
    # 5. Non-monotonic count
    print("\n5. NON-MONOTONIC RELATIONSHIPS")
    nonmono = results['nonmonotonic']
    if len(nonmono) > 0:
        shapes = nonmono['shape'].value_counts()
        for shape, count in shapes.items():
            print(f"   {shape}: {count}")


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "#"*80)
    print("# COMPREHENSIVE ANALYSIS (Excluding Embedding Attack)")
    print(f"# Generated: {timestamp}")
    print("#"*80)
    
    # Load experiments
    print("\n" + "="*80)
    print("LOADING EXPERIMENTS")
    print("="*80)
    
    experiments = load_all_experiments()
    
    for name, data in experiments.items():
        attacks = list(data['attacks'].keys())
        print(f"  [OK] {name}: {attacks}")
    
    print(f"\nLoaded {len(experiments)} experiments")
    
    if len(experiments) == 0:
        print("No experiments found!")
        return
    
    # Run all analyses
    results = {}
    
    results['summary'] = analyze_summary(experiments)
    results['correlations'] = analyze_correlations(experiments)
    results['single_feature'] = analyze_single_feature(experiments)
    results['predictions'] = analyze_multi_feature(experiments)
    results['tree_rules'] = analyze_tree_rules(experiments)
    results['cross_experiment'] = analyze_cross_experiment(results['correlations'])
    results['cross_attack'] = analyze_cross_attack(results['correlations'])
    results['geometry_change'] = analyze_geometry_change(experiments)
    results['nonmonotonic'] = analyze_nonmonotonic(experiments)
    results['importance'] = analyze_permutation_importance(experiments)
    results['feature_groups'] = analyze_feature_groups(experiments)
    
    # Key findings
    generate_key_findings(results)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_dir = Path('./comprehensive_analysis')
    output_dir.mkdir(exist_ok=True)
    
    for name, df in results.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            path = output_dir / f'{name}.csv'
            df.to_csv(path, index=False)
            print(f"  [OK] {path}")
    
    print("\n" + "#"*80)
    print("# ANALYSIS COMPLETE")
    print(f"# Results saved to: {output_dir}")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
