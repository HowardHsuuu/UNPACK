"""
Nonlinear Relationship Analysis
Explores threshold effects, interactions, and non-monotonic relationships
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
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


def load_experiment(name, path):
    path = Path(path)
    if not path.exists():
        return None
    
    data = {'name': name}
    files = {
        'base_features': 'base_geometric_features.csv',
        'unlearned_features': 'unlearned_geometric_features.csv',
        'retention': 'direct_retention.csv',
        'attack_results': 'attack_results.csv',
        'embedding_attack': 'embedding_attack_results.csv',
    }
    
    for key, filename in files.items():
        filepath = path / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
    
    if 'base_features' not in data or 'retention' not in data:
        return None
    
    return data


def prepare_data(data, geometry='base'):
    """Prepare X, y for modeling"""
    retention = data['retention'].copy()
    retention['unlearning_success'] = 1 - retention['retention']
    
    if 'attack_results' in data:
        attack_df = data['attack_results']
    elif 'embedding_attack' in data:
        attack_df = data['embedding_attack']
    else:
        attack_df = retention.copy()
        attack_df['caf'] = retention['retention']
    
    outcomes = retention[['query_id', 'unlearning_success']].merge(
        attack_df[['query_id', 'caf']], on='query_id', how='left'
    )
    outcomes['caf'] = outcomes['caf'].fillna(0)
    
    if geometry == 'base':
        feat_df = data['base_features']
    else:
        feat_df = data.get('unlearned_features')
        if feat_df is None:
            return None, None, None
    
    merged = feat_df.merge(outcomes, on='query_id')
    available = [f for f in ALL_FEATURES if f in merged.columns]
    
    X = merged[available].values
    y_unlearn = merged['unlearning_success'].values
    y_extract = merged['caf'].values
    
    # Remove NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_unlearn) | np.isnan(y_extract))
    
    return X[mask], y_unlearn[mask], y_extract[mask], available


# ============================================================
# ANALYSIS 1: Decision Tree Rules
# ============================================================

def analyze_decision_tree_rules(experiments):
    """Extract interpretable rules from decision trees"""
    print("\n" + "="*70)
    print("ANALYSIS 1: DECISION TREE RULES")
    print("Extracting interpretable threshold rules")
    print("="*70)
    
    all_rules = []
    
    for exp_name, data in experiments.items():
        for geometry in ['base', 'unlearned']:
            result = prepare_data(data, geometry)
            if result[0] is None:
                continue
            
            X, y_unlearn, y_extract, features = result
            
            if len(X) < 20:
                continue
            
            for target_name, y in [('Unlearning', y_unlearn), ('Extraction', y_extract)]:
                # Shallow tree for interpretability
                tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=42)
                tree.fit(X, y)
                
                # Get CV score
                cv_scores = cross_val_score(tree, X, y, cv=5, scoring='r2')
                
                if cv_scores.mean() > -0.5:  # Only show if somewhat predictive
                    print(f"\n{exp_name} / {geometry} -> {target_name}")
                    print(f"CV R2: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
                    print("-" * 50)
                    
                    # Extract rules
                    tree_rules = export_text(tree, feature_names=features, max_depth=3)
                    print(tree_rules)
                    
                    # Store important thresholds
                    tree_struct = tree.tree_
                    for node_id in range(tree_struct.node_count):
                        if tree_struct.feature[node_id] != -2:  # Not a leaf
                            feat_idx = tree_struct.feature[node_id]
                            threshold = tree_struct.threshold[node_id]
                            all_rules.append({
                                'experiment': exp_name,
                                'geometry': geometry,
                                'target': target_name,
                                'feature': features[feat_idx],
                                'threshold': threshold,
                                'cv_r2': cv_scores.mean()
                            })
    
    return pd.DataFrame(all_rules)


# ============================================================
# ANALYSIS 2: Threshold Detection
# ============================================================

def analyze_thresholds(experiments):
    """Find optimal thresholds for each feature"""
    print("\n" + "="*70)
    print("ANALYSIS 2: THRESHOLD DETECTION")
    print("Finding optimal split points for each feature")
    print("="*70)
    
    all_thresholds = []
    
    for exp_name, data in experiments.items():
        for geometry in ['base', 'unlearned']:
            result = prepare_data(data, geometry)
            if result[0] is None:
                continue
            
            X, y_unlearn, y_extract, features = result
            
            if len(X) < 20:
                continue
            
            for target_name, y in [('Unlearning', y_unlearn), ('Extraction', y_extract)]:
                for feat_idx, feat_name in enumerate(features):
                    x = X[:, feat_idx]
                    
                    # Try different thresholds
                    percentiles = [25, 50, 75]
                    best_diff = 0
                    best_threshold = None
                    best_pct = None
                    
                    for pct in percentiles:
                        threshold = np.percentile(x, pct)
                        
                        low_mask = x <= threshold
                        high_mask = x > threshold
                        
                        if low_mask.sum() < 5 or high_mask.sum() < 5:
                            continue
                        
                        low_mean = y[low_mask].mean()
                        high_mean = y[high_mask].mean()
                        diff = high_mean - low_mean
                        
                        if abs(diff) > abs(best_diff):
                            best_diff = diff
                            best_threshold = threshold
                            best_pct = pct
                    
                    if best_threshold is not None and abs(best_diff) > 0.05:
                        all_thresholds.append({
                            'experiment': exp_name,
                            'geometry': geometry,
                            'target': target_name,
                            'feature': feat_name,
                            'threshold': best_threshold,
                            'percentile': best_pct,
                            'diff': best_diff,
                            'direction': 'higher is better' if best_diff > 0 else 'lower is better'
                        })
    
    df = pd.DataFrame(all_thresholds)
    
    if len(df) > 0:
        # Show top thresholds
        df['abs_diff'] = df['diff'].abs()
        df = df.sort_values('abs_diff', ascending=False)
        
        print("\nTop threshold effects (|diff| > 0.1):")
        print("-" * 100)
        print(f"{'Experiment':<18} {'Geom':<10} {'Target':<12} {'Feature':<25} {'Threshold':>10} {'Diff':>8} {'Direction'}")
        print("-" * 100)
        
        for _, row in df[df['abs_diff'] > 0.1].head(20).iterrows():
            print(f"{row['experiment']:<18} {row['geometry']:<10} {row['target']:<12} "
                  f"{row['feature']:<25} {row['threshold']:>10.3f} {row['diff']:>+8.3f} {row['direction']}")
    
    return df


# ============================================================
# ANALYSIS 3: Feature Interactions
# ============================================================

def analyze_interactions(experiments):
    """Detect feature interactions using polynomial features"""
    print("\n" + "="*70)
    print("ANALYSIS 3: FEATURE INTERACTIONS")
    print("Testing if feature combinations improve prediction")
    print("="*70)
    
    all_interactions = []
    
    for exp_name, data in experiments.items():
        for geometry in ['base', 'unlearned']:
            result = prepare_data(data, geometry)
            if result[0] is None:
                continue
            
            X, y_unlearn, y_extract, features = result
            
            if len(X) < 30:
                continue
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            for target_name, y in [('Unlearning', y_unlearn), ('Extraction', y_extract)]:
                # Linear only
                lr = LinearRegression()
                cv_linear = cross_val_score(lr, X_scaled, y, cv=5, scoring='r2')
                
                # Polynomial (interactions only, degree 2)
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_poly = poly.fit_transform(X_scaled)
                
                lr_poly = LinearRegression()
                cv_poly = cross_val_score(lr_poly, X_poly, y, cv=5, scoring='r2')
                
                improvement = cv_poly.mean() - cv_linear.mean()
                
                all_interactions.append({
                    'experiment': exp_name,
                    'geometry': geometry,
                    'target': target_name,
                    'linear_r2': cv_linear.mean(),
                    'interaction_r2': cv_poly.mean(),
                    'improvement': improvement,
                    'n_features': X.shape[1],
                    'n_interaction_features': X_poly.shape[1]
                })
                
                if improvement > 0.05:
                    print(f"\n{exp_name} / {geometry} -> {target_name}")
                    print(f"  Linear R2:      {cv_linear.mean():.3f}")
                    print(f"  Interaction R2: {cv_poly.mean():.3f}")
                    print(f"  Improvement:    {improvement:+.3f} ***")
    
    df = pd.DataFrame(all_interactions)
    
    print("\n\nSummary: Interaction Effects")
    print("-" * 80)
    print(f"{'Experiment':<18} {'Geom':<10} {'Target':<12} {'Linear':>10} {'Interact':>10} {'Improve':>10}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        marker = "***" if row['improvement'] > 0.05 else ""
        print(f"{row['experiment']:<18} {row['geometry']:<10} {row['target']:<12} "
              f"{row['linear_r2']:>10.3f} {row['interaction_r2']:>10.3f} {row['improvement']:>+10.3f} {marker}")
    
    return df


# ============================================================
# ANALYSIS 4: Permutation Importance (Non-linear)
# ============================================================

def analyze_permutation_importance(experiments):
    """Use permutation importance with RandomForest to find important features"""
    print("\n" + "="*70)
    print("ANALYSIS 4: PERMUTATION IMPORTANCE (RandomForest)")
    print("Finding important features for non-linear models")
    print("="*70)
    
    all_importance = []
    
    for exp_name, data in experiments.items():
        for geometry in ['base', 'unlearned']:
            result = prepare_data(data, geometry)
            if result[0] is None:
                continue
            
            X, y_unlearn, y_extract, features = result
            
            if len(X) < 30:
                continue
            
            for target_name, y in [('Unlearning', y_unlearn), ('Extraction', y_extract)]:
                # Train RandomForest
                rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                rf.fit(X, y)
                
                # CV score
                cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
                
                if cv_scores.mean() < -0.5:
                    continue
                
                # Permutation importance
                perm_imp = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
                
                print(f"\n{exp_name} / {geometry} -> {target_name} (CV R2: {cv_scores.mean():.3f})")
                print("-" * 50)
                
                # Sort by importance
                sorted_idx = perm_imp.importances_mean.argsort()[::-1]
                
                for i in sorted_idx[:5]:
                    imp = perm_imp.importances_mean[i]
                    std = perm_imp.importances_std[i]
                    
                    if imp > 0.001:  # Only show meaningful
                        print(f"  {features[i]:<35} {imp:.4f} +/- {std:.4f}")
                        
                        all_importance.append({
                            'experiment': exp_name,
                            'geometry': geometry,
                            'target': target_name,
                            'feature': features[i],
                            'importance': imp,
                            'importance_std': std,
                            'cv_r2': cv_scores.mean()
                        })
    
    return pd.DataFrame(all_importance)


# ============================================================
# ANALYSIS 5: Binned Analysis (Non-monotonic check)
# ============================================================

def analyze_binned_relationships(experiments):
    """Check for non-monotonic relationships using bins"""
    print("\n" + "="*70)
    print("ANALYSIS 5: BINNED ANALYSIS (Non-monotonic check)")
    print("Checking if relationship is U-shaped or inverted-U")
    print("="*70)
    
    all_binned = []
    
    for exp_name, data in experiments.items():
        for geometry in ['base', 'unlearned']:
            result = prepare_data(data, geometry)
            if result[0] is None:
                continue
            
            X, y_unlearn, y_extract, features = result
            
            if len(X) < 30:
                continue
            
            for target_name, y in [('Unlearning', y_unlearn), ('Extraction', y_extract)]:
                for feat_idx, feat_name in enumerate(features):
                    x = X[:, feat_idx]
                    
                    # Create 4 bins
                    try:
                        bins = pd.qcut(x, q=4, duplicates='drop')
                        bin_means = pd.Series(y).groupby(bins).mean()
                        
                        if len(bin_means) < 3:
                            continue
                        
                        # Check for non-monotonicity
                        values = bin_means.values
                        
                        # Is it monotonic increasing?
                        monotonic_inc = all(values[i] <= values[i+1] for i in range(len(values)-1))
                        # Is it monotonic decreasing?
                        monotonic_dec = all(values[i] >= values[i+1] for i in range(len(values)-1))
                        
                        if not monotonic_inc and not monotonic_dec:
                            # Non-monotonic!
                            # Check shape
                            mid = len(values) // 2
                            if values[0] > values[mid] < values[-1]:
                                shape = "U-shaped"
                            elif values[0] < values[mid] > values[-1]:
                                shape = "Inverted-U"
                            else:
                                shape = "Complex"
                            
                            range_y = values.max() - values.min()
                            
                            if range_y > 0.05:  # Meaningful variation
                                all_binned.append({
                                    'experiment': exp_name,
                                    'geometry': geometry,
                                    'target': target_name,
                                    'feature': feat_name,
                                    'shape': shape,
                                    'bin_values': str(list(values.round(3))),
                                    'range': range_y
                                })
                                
                    except Exception:
                        continue
    
    df = pd.DataFrame(all_binned)
    
    if len(df) > 0:
        df = df.sort_values('range', ascending=False)
        
        print("\nNon-monotonic relationships detected:")
        print("-" * 100)
        print(f"{'Experiment':<18} {'Geom':<10} {'Target':<12} {'Feature':<25} {'Shape':<12} {'Range':>8}")
        print("-" * 100)
        
        for _, row in df.head(15).iterrows():
            print(f"{row['experiment']:<18} {row['geometry']:<10} {row['target']:<12} "
                  f"{row['feature']:<25} {row['shape']:<12} {row['range']:>8.3f}")
            print(f"    Bin means: {row['bin_values']}")
    else:
        print("\nNo strong non-monotonic relationships detected.")
    
    return df


# ============================================================
# ANALYSIS 6: Model Comparison Summary
# ============================================================

def compare_models(experiments):
    """Compare linear vs non-linear models"""
    print("\n" + "="*70)
    print("ANALYSIS 6: LINEAR vs NON-LINEAR MODEL COMPARISON")
    print("="*70)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        for geometry in ['base', 'unlearned']:
            result = prepare_data(data, geometry)
            if result[0] is None:
                continue
            
            X, y_unlearn, y_extract, features = result
            
            if len(X) < 30:
                continue
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            for target_name, y in [('Unlearning', y_unlearn), ('Extraction', y_extract)]:
                models = {
                    'Linear': LinearRegression(),
                    'DecisionTree': DecisionTreeRegressor(max_depth=4, random_state=42),
                    'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42),
                    'GradientBoosting': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
                }
                
                for model_name, model in models.items():
                    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                    
                    all_results.append({
                        'experiment': exp_name,
                        'geometry': geometry,
                        'target': target_name,
                        'model': model_name,
                        'cv_r2_mean': cv_scores.mean(),
                        'cv_r2_std': cv_scores.std()
                    })
    
    df = pd.DataFrame(all_results)
    
    # Pivot table
    print("\nModel Comparison (CV R2):")
    print("-" * 90)
    
    pivot = df.pivot_table(
        values='cv_r2_mean',
        index=['experiment', 'geometry', 'target'],
        columns='model'
    )
    
    # Add column for nonlinear advantage
    pivot['NonLinear_Advantage'] = pivot[['RandomForest', 'GradientBoosting']].max(axis=1) - pivot['Linear']
    
    print(pivot.round(3).to_string())
    
    # Summary
    print("\n\nKey Insight:")
    print("-" * 50)
    
    advantages = pivot['NonLinear_Advantage']
    strong_nonlinear = advantages[advantages > 0.1]
    
    if len(strong_nonlinear) > 0:
        print(f"Cases where non-linear models outperform linear by >0.1 R2:")
        for idx, val in strong_nonlinear.items():
            print(f"  {idx}: +{val:.3f}")
    else:
        print("No strong non-linear advantages detected.")
    
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "#"*70)
    print("# NON-LINEAR RELATIONSHIP ANALYSIS")
    print("# Exploring thresholds, interactions, and non-monotonic effects")
    print("#"*70)
    
    # Load experiments
    print("\n" + "="*70)
    print("LOADING EXPERIMENTS")
    print("="*70)
    
    experiments = {}
    for name, path in EXPERIMENTS.items():
        data = load_experiment(name, path)
        if data:
            experiments[name] = data
            print(f"  [OK] {name}")
        else:
            print(f"  [--] {name}: not found")
    
    if len(experiments) == 0:
        print("No experiments found!")
        return
    
    # Run analyses
    results = {}
    
    results['tree_rules'] = analyze_decision_tree_rules(experiments)
    results['thresholds'] = analyze_thresholds(experiments)
    results['interactions'] = analyze_interactions(experiments)
    results['permutation_importance'] = analyze_permutation_importance(experiments)
    results['binned'] = analyze_binned_relationships(experiments)
    results['model_comparison'] = compare_models(experiments)
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    output_dir = Path('./nonlinear_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    for name, df in results.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            path = output_dir / f'{name}.csv'
            df.to_csv(path, index=False)
            print(f"  [OK] {path}")
    
    # Final summary
    print("\n" + "#"*70)
    print("# SUMMARY: NON-LINEAR INSIGHTS")
    print("#"*70)
    
    # Check model comparison
    if len(results['model_comparison']) > 0:
        mc = results['model_comparison']
        linear = mc[mc['model'] == 'Linear']['cv_r2_mean'].mean()
        rf = mc[mc['model'] == 'RandomForest']['cv_r2_mean'].mean()
        print(f"\n1. MODEL PERFORMANCE:")
        print(f"   Average Linear R2:       {linear:.3f}")
        print(f"   Average RandomForest R2: {rf:.3f}")
        print(f"   Non-linear advantage:    {rf - linear:+.3f}")
    
    # Threshold effects
    if len(results['thresholds']) > 0:
        top_thresh = results['thresholds'].nlargest(3, 'abs_diff')
        print(f"\n2. STRONGEST THRESHOLD EFFECTS:")
        for _, row in top_thresh.iterrows():
            print(f"   {row['feature']}: {row['direction']} (diff={row['diff']:+.3f})")
    
    # Non-monotonic
    if len(results['binned']) > 0:
        print(f"\n3. NON-MONOTONIC RELATIONSHIPS: {len(results['binned'])} detected")
        shapes = results['binned']['shape'].value_counts()
        for shape, count in shapes.items():
            print(f"   {shape}: {count}")
    
    print("\n" + "#"*70)
    print("# ANALYSIS COMPLETE")
    print(f"# Results saved to: {output_dir}")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
