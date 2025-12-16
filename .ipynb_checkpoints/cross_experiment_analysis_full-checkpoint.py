"""
Comprehensive Cross-Experiment Analysis
Tests ALL features individually and in combinations
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
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

FEATURE_GROUPS = {
    'density': ['local_density_mean', 'local_density_std', 'local_density_max', 'local_density_min'],
    'separability': ['separability_mean', 'separability_std', 'separability_max', 'separability_min'],
    'centrality': ['centrality_mean', 'centrality_std', 'centrality_max', 'centrality_min'],
    'isolation': ['isolation_mean', 'isolation_std', 'isolation_max', 'isolation_min'],
    'compactness': ['cluster_compactness_mean', 'cluster_compactness_std', 'cluster_compactness_max', 'cluster_compactness_min'],
    'consistency': ['cross_layer_consistency_mean', 'cross_layer_consistency_std', 'cross_layer_consistency_max', 'cross_layer_consistency_min'],
}


def load_experiment(name, path):
    path = Path(path)
    if not path.exists():
        return None
    
    data = {'name': name}
    files = {
        'base_features': 'base_geometric_features.csv',
        'unlearned_features': 'unlearned_geometric_features.csv',
        'retention': 'direct_retention.csv',
        'base_accuracy': 'base_accuracy.csv',
        'attack_results': 'attack_results.csv',
        'embedding_attack': 'embedding_attack_results.csv',
        'activation_steering': 'activation_steering_results.csv',
        'prompt_attack': 'prompt_attack_results.csv',
    }
    
    for key, filename in files.items():
        filepath = path / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
    
    if 'base_features' not in data or 'retention' not in data:
        return None
    
    return data


def load_all_experiments():
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
            print(f"  [--] {name}: not found, skipping")
    
    print(f"\nLoaded {len(experiments)} experiments")
    return experiments


def prepare_experiment_data(data):
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
    
    return outcomes


def analyze_all_feature_correlations(experiments):
    print("\n" + "="*70)
    print("ANALYSIS 1: ALL FEATURES CORRELATION")
    print("="*70)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_experiment_data(data)
        available_features = [f for f in ALL_FEATURES if f in data['base_features'].columns]
        
        for geom_name, feat_df in [('base', data['base_features']), 
                                    ('unlearned', data.get('unlearned_features'))]:
            if feat_df is None:
                continue
            
            merged = feat_df.merge(outcomes, on='query_id')
            
            for target_col, target_name in [('unlearning_success', 'Unlearning'), 
                                             ('caf', 'Extraction')]:
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
                        'significant_pearson': p_pearson < 0.05,
                        'significant_spearman': p_spearman < 0.05,
                        'direction': 'positive' if r_pearson > 0 else 'negative',
                        'n_samples': len(x_clean)
                    })
    
    df = pd.DataFrame(all_results)
    
    sig = df[df['significant_pearson']]
    print(f"\nTotal correlations computed: {len(df)}")
    print(f"Significant (p < 0.05): {len(sig)} ({len(sig)/len(df)*100:.1f}%)")
    
    return df


def single_feature_prediction_all(experiments):
    print("\n" + "="*70)
    print("ANALYSIS 2: SINGLE FEATURE PREDICTION (ALL FEATURES)")
    print("="*70)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_experiment_data(data)
        available_features = [f for f in ALL_FEATURES if f in data['base_features'].columns]
        
        for geom_name, feat_df in [('base', data['base_features']), 
                                    ('unlearned', data.get('unlearned_features'))]:
            if feat_df is None:
                continue
            
            merged = feat_df.merge(outcomes, on='query_id')
            
            for target_col, target_name in [('unlearning_success', 'Unlearning'), 
                                             ('caf', 'Extraction')]:
                for feat in available_features:
                    if feat not in merged.columns:
                        continue
                    
                    X = merged[feat].values.reshape(-1, 1)
                    y = merged[target_col].values
                    
                    mask = ~(np.isnan(X.flatten()) | np.isnan(y))
                    X, y = X[mask], y[mask]
                    
                    if len(X) < 15:
                        continue
                    
                    r, p = pearsonr(X.flatten(), y)
                    
                    model_lr = LinearRegression()
                    cv_lr = cross_val_score(model_lr, X, y, cv=5, scoring='r2')
                    
                    model_lr.fit(X, y)
                    coef = model_lr.coef_[0]
                    
                    all_results.append({
                        'experiment': exp_name,
                        'geometry': geom_name,
                        'target': target_name,
                        'feature': feat,
                        'pearson_r': r,
                        'abs_r': abs(r),
                        'p_value': p,
                        'significant': p < 0.05,
                        'cv_r2_mean': cv_lr.mean(),
                        'cv_r2_std': cv_lr.std(),
                        'coefficient': coef,
                        'direction': 'positive' if coef > 0 else 'negative',
                        'n_samples': len(X)
                    })
    
    df = pd.DataFrame(all_results)
    
    print("\nBest predictive features (by |r|):")
    print("-" * 80)
    
    for exp_name in df['experiment'].unique():
        print(f"\n{exp_name}:")
        for target in ['Unlearning', 'Extraction']:
            subset = df[(df['experiment'] == exp_name) & (df['target'] == target)]
            if len(subset) == 0:
                continue
            
            # Fixed: use abs_r column instead of key argument
            top3 = subset.nlargest(3, 'abs_r')
            
            print(f"  {target}:")
            for _, row in top3.iterrows():
                sig = "*" if row['significant'] else ""
                dir_symbol = "+" if row['direction'] == 'positive' else "-"
                print(f"    {dir_symbol} {row['feature']}: r={row['pearson_r']:.3f}{sig}, CV R2={row['cv_r2_mean']:.3f}")
    
    return df


def analyze_feature_groups(experiments):
    print("\n" + "="*70)
    print("ANALYSIS 3: FEATURE GROUP ANALYSIS")
    print("="*70)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_experiment_data(data)
        
        for geom_name, feat_df in [('base', data['base_features']), 
                                    ('unlearned', data.get('unlearned_features'))]:
            if feat_df is None:
                continue
            
            merged = feat_df.merge(outcomes, on='query_id')
            
            for target_col, target_name in [('unlearning_success', 'Unlearning'), 
                                             ('caf', 'Extraction')]:
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
                    
                    models = {
                        'LinearRegression': LinearRegression(),
                        'Ridge': Ridge(alpha=1.0),
                    }
                    
                    for model_name, model in models.items():
                        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                        
                        all_results.append({
                            'experiment': exp_name,
                            'geometry': geom_name,
                            'target': target_name,
                            'feature_group': group_name,
                            'model': model_name,
                            'n_features': len(available),
                            'cv_r2_mean': cv_scores.mean(),
                            'cv_r2_std': cv_scores.std(),
                            'n_samples': len(X)
                        })
    
    df = pd.DataFrame(all_results)
    
    print("\nFeature Group Performance (CV R2, LinearRegression):")
    print("-" * 90)
    
    lr_results = df[df['model'] == 'LinearRegression']
    if len(lr_results) > 0:
        pivot = lr_results.pivot_table(
            values='cv_r2_mean', 
            index=['experiment', 'geometry'], 
            columns=['target', 'feature_group'],
            aggfunc='mean'
        )
        print(pivot.round(3).to_string())
    
    return df


def analyze_cross_experiment_consistency(corr_df):
    print("\n" + "="*70)
    print("ANALYSIS 4: CROSS-EXPERIMENT CONSISTENCY")
    print("="*70)
    
    sig = corr_df[corr_df['significant_pearson']].copy()
    
    if len(sig) == 0:
        print("No significant correlations found")
        return pd.DataFrame()
    
    consistency = sig.groupby(['feature', 'geometry', 'target']).agg({
        'pearson_r': ['mean', 'std', 'count'],
        'direction': lambda x: x.mode()[0] if len(x) > 0 else 'mixed',
        'experiment': lambda x: list(x.unique())
    }).reset_index()
    
    consistency.columns = ['feature', 'geometry', 'target', 'mean_r', 'std_r', 
                          'n_experiments', 'main_direction', 'experiments']
    
    consistent = consistency[consistency['n_experiments'] >= 2].sort_values(
        'n_experiments', ascending=False
    )
    
    print("\nFeatures significant in 2+ experiments:")
    print("-" * 90)
    print(f"{'Feature':<30} {'Geom':<10} {'Target':<12} {'n':>3} {'Mean r':>8} {'Dir':<8} {'Experiments'}")
    print("-" * 90)
    
    for _, row in consistent.iterrows():
        exp_str = ', '.join(row['experiments'])[:25]
        dir_symbol = "+" if row['main_direction'] == 'positive' else "-"
        print(f"{row['feature']:<30} {row['geometry']:<10} {row['target']:<12} "
              f"{row['n_experiments']:>3} {row['mean_r']:>8.3f} {dir_symbol:<8} {exp_str}")
    
    return consistent


def analyze_base_vs_unlearned(experiments):
    print("\n" + "="*70)
    print("ANALYSIS 5: BASE vs UNLEARNED GEOMETRY COMPARISON")
    print("="*70)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        if 'unlearned_features' not in data:
            continue
        
        base_feat = data['base_features']
        unlearned_feat = data['unlearned_features']
        
        merged = base_feat.merge(unlearned_feat, on='query_id', suffixes=('_base', '_unlearned'))
        
        available_features = [f for f in ALL_FEATURES if f in base_feat.columns]
        
        for feat in available_features:
            base_col = f"{feat}_base"
            unlearned_col = f"{feat}_unlearned"
            
            if base_col not in merged.columns or unlearned_col not in merged.columns:
                continue
            
            base_vals = merged[base_col].dropna()
            unlearned_vals = merged[unlearned_col].dropna()
            
            if len(base_vals) < 10 or len(unlearned_vals) < 10:
                continue
            
            t_stat, t_pval = ttest_ind(base_vals, unlearned_vals)
            u_stat, u_pval = mannwhitneyu(base_vals, unlearned_vals, alternative='two-sided')
            
            pooled_std = np.sqrt((base_vals.std()**2 + unlearned_vals.std()**2) / 2)
            cohens_d = (base_vals.mean() - unlearned_vals.mean()) / pooled_std if pooled_std > 0 else 0
            
            all_results.append({
                'experiment': exp_name,
                'feature': feat,
                'base_mean': base_vals.mean(),
                'unlearned_mean': unlearned_vals.mean(),
                'diff_mean': base_vals.mean() - unlearned_vals.mean(),
                'diff_pct': (base_vals.mean() - unlearned_vals.mean()) / base_vals.mean() * 100 if base_vals.mean() != 0 else 0,
                't_statistic': t_stat,
                't_pvalue': t_pval,
                'u_pvalue': u_pval,
                'cohens_d': cohens_d,
                'abs_cohens_d': abs(cohens_d),
                'significant': t_pval < 0.05
            })
    
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("No comparison data available")
        return df
    
    sig = df[df['significant']].sort_values('abs_cohens_d', ascending=False)
    
    print("\nSignificant changes after unlearning (|Cohen's d| > 0.2):")
    print("-" * 100)
    print(f"{'Experiment':<18} {'Feature':<30} {'Base':>10} {'Unlearned':>10} {'Diff%':>8} {'Cohen d':>10}")
    print("-" * 100)
    
    for _, row in sig[sig['abs_cohens_d'] > 0.2].iterrows():
        print(f"{row['experiment']:<18} {row['feature']:<30} "
              f"{row['base_mean']:>10.3f} {row['unlearned_mean']:>10.3f} "
              f"{row['diff_pct']:>7.1f}% {row['cohens_d']:>10.3f}")
    
    return df


def analyze_multi_feature_models(experiments):
    print("\n" + "="*70)
    print("ANALYSIS 6: MULTI-FEATURE MODELS")
    print("="*70)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_experiment_data(data)
        
        for geom_name, feat_df in [('base', data['base_features']), 
                                    ('unlearned', data.get('unlearned_features'))]:
            if feat_df is None:
                continue
            
            merged = feat_df.merge(outcomes, on='query_id')
            available_features = [f for f in ALL_FEATURES if f in merged.columns]
            
            if len(available_features) < 3:
                continue
            
            for target_col, target_name in [('unlearning_success', 'Unlearning'), 
                                             ('caf', 'Extraction')]:
                X = merged[available_features].values
                y = merged[target_col].values
                
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X, y = X[mask], y[mask]
                
                if len(X) < 20:
                    continue
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                models = {
                    'Ridge': Ridge(alpha=1.0),
                    'Lasso': Lasso(alpha=0.1),
                    'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
                }
                
                for model_name, model in models.items():
                    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                    
                    model.fit(X_scaled, y)
                    
                    if hasattr(model, 'coef_'):
                        importances = dict(zip(available_features, model.coef_))
                        top_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    elif hasattr(model, 'feature_importances_'):
                        importances = dict(zip(available_features, model.feature_importances_))
                        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                    else:
                        top_features = []
                    
                    all_results.append({
                        'experiment': exp_name,
                        'geometry': geom_name,
                        'target': target_name,
                        'model': model_name,
                        'n_features': len(available_features),
                        'cv_r2_mean': cv_scores.mean(),
                        'cv_r2_std': cv_scores.std(),
                        'top_features': str(top_features),
                        'n_samples': len(X)
                    })
    
    df = pd.DataFrame(all_results)
    
    print("\nMulti-feature model performance:")
    print("-" * 90)
    print(f"{'Experiment':<18} {'Geom':<10} {'Target':<12} {'Model':<15} {'CV R2':>12} {'n_feat':>8}")
    print("-" * 90)
    
    for _, row in df.sort_values(['experiment', 'geometry', 'target', 'model']).iterrows():
        print(f"{row['experiment']:<18} {row['geometry']:<10} {row['target']:<12} "
              f"{row['model']:<15} {row['cv_r2_mean']:>8.3f}+/-{row['cv_r2_std']:.3f} {row['n_features']:>8}")
    
    return df


def analyze_mutual_information(experiments):
    print("\n" + "="*70)
    print("ANALYSIS 7: MUTUAL INFORMATION (Non-linear relationships)")
    print("="*70)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        outcomes = prepare_experiment_data(data)
        
        for geom_name, feat_df in [('base', data['base_features']), 
                                    ('unlearned', data.get('unlearned_features'))]:
            if feat_df is None:
                continue
            
            merged = feat_df.merge(outcomes, on='query_id')
            available_features = [f for f in ALL_FEATURES if f in merged.columns]
            
            for target_col, target_name in [('unlearning_success', 'Unlearning'), 
                                             ('caf', 'Extraction')]:
                X = merged[available_features].values
                y = merged[target_col].values
                
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X, y = X[mask], y[mask]
                
                if len(X) < 20:
                    continue
                
                mi_scores = mutual_info_regression(X, y, random_state=42)
                
                for feat, mi in zip(available_features, mi_scores):
                    all_results.append({
                        'experiment': exp_name,
                        'geometry': geom_name,
                        'target': target_name,
                        'feature': feat,
                        'mutual_info': mi,
                        'n_samples': len(X)
                    })
    
    df = pd.DataFrame(all_results)
    
    print("\nTop 5 features by Mutual Information:")
    print("-" * 80)
    
    for exp_name in df['experiment'].unique():
        print(f"\n{exp_name}:")
        for target in ['Unlearning', 'Extraction']:
            for geom in ['base', 'unlearned']:
                subset = df[(df['experiment'] == exp_name) & 
                           (df['target'] == target) & 
                           (df['geometry'] == geom)]
                if len(subset) == 0:
                    continue
                
                top5 = subset.nlargest(5, 'mutual_info')
                print(f"  {geom} -> {target}:")
                for _, row in top5.iterrows():
                    print(f"    {row['feature']}: MI={row['mutual_info']:.4f}")
    
    return df


def compute_summary_statistics(experiments):
    print("\n" + "="*70)
    print("ANALYSIS 8: SUMMARY STATISTICS")
    print("="*70)
    
    all_results = []
    
    for exp_name, data in experiments.items():
        row = {'experiment': exp_name}
        
        row['n_queries'] = len(data['base_features'])
        
        if 'base_accuracy' in data:
            row['base_caf_mean'] = data['base_accuracy']['base_caf'].mean()
            row['base_caf_std'] = data['base_accuracy']['base_caf'].std()
        
        if 'retention' in data:
            row['retention_mean'] = data['retention']['retention'].mean()
            row['retention_std'] = data['retention']['retention'].std()
            row['unlearning_effectiveness'] = 1 - row['retention_mean']
        
        for attack_name, attack_key in [('embedding', 'embedding_attack'), 
                                         ('steering', 'activation_steering'),
                                         ('prompt', 'prompt_attack')]:
            if attack_key in data:
                df_attack = data[attack_key]
                caf_col = 'caf' if 'caf' in df_attack.columns else 'best_caf'
                if caf_col in df_attack.columns:
                    row[f'{attack_name}_caf_mean'] = df_attack[caf_col].mean()
                    row[f'{attack_name}_caf_std'] = df_attack[caf_col].std()
                    row[f'{attack_name}_success_rate'] = (df_attack[caf_col] > 0).mean()
        
        all_results.append(row)
    
    df = pd.DataFrame(all_results)
    
    print("\nExperiment Summary:")
    print("-" * 120)
    cols = ['experiment', 'n_queries', 'base_caf_mean', 'retention_mean', 
            'unlearning_effectiveness', 'embedding_caf_mean', 'steering_caf_mean', 'prompt_caf_mean']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    
    return df


def main():
    print("\n" + "#"*70)
    print("# COMPREHENSIVE CROSS-EXPERIMENT ANALYSIS")
    print("# Testing ALL features and combinations")
    print("#"*70)
    
    experiments = load_all_experiments()
    
    if len(experiments) == 0:
        print("No experiments found!")
        return
    
    results = {}
    
    results['correlations'] = analyze_all_feature_correlations(experiments)
    results['single_feature'] = single_feature_prediction_all(experiments)
    results['feature_groups'] = analyze_feature_groups(experiments)
    results['consistency'] = analyze_cross_experiment_consistency(results['correlations'])
    results['base_vs_unlearned'] = analyze_base_vs_unlearned(experiments)
    results['multi_feature'] = analyze_multi_feature_models(experiments)
    results['mutual_info'] = analyze_mutual_information(experiments)
    results['summary'] = compute_summary_statistics(experiments)
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    output_dir = Path('./cross_experiment_results_full')
    output_dir.mkdir(exist_ok=True)
    
    for name, df in results.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            path = output_dir / f'{name}.csv'
            df.to_csv(path, index=False)
            print(f"  [OK] {path}")
    
    print("\n" + "#"*70)
    print("# KEY FINDINGS SUMMARY")
    print("#"*70)
    
    sig_corr = results['correlations'][results['correlations']['significant_pearson']]
    if len(sig_corr) > 0:
        print(f"\n1. SIGNIFICANT CORRELATIONS: {len(sig_corr)} found")
        
        feat_counts = sig_corr.groupby('feature').size().sort_values(ascending=False)
        print(f"   Most consistent features:")
        for feat, count in feat_counts.head(5).items():
            avg_r = sig_corr[sig_corr['feature'] == feat]['pearson_r'].mean()
            print(f"     - {feat}: {count} significant correlations, avg r = {avg_r:.3f}")
    
    if len(results['multi_feature']) > 0:
        best = results['multi_feature'].loc[results['multi_feature']['cv_r2_mean'].idxmax()]
        print(f"\n2. BEST MULTI-FEATURE MODEL:")
        print(f"   {best['experiment']} / {best['geometry']} -> {best['target']}")
        print(f"   Model: {best['model']}, CV R2 = {best['cv_r2_mean']:.3f}")
    
    if len(results['base_vs_unlearned']) > 0:
        sig_changes = results['base_vs_unlearned'][
            (results['base_vs_unlearned']['significant']) & 
            (results['base_vs_unlearned']['abs_cohens_d'] > 0.5)
        ]
        if len(sig_changes) > 0:
            print(f"\n3. LARGE GEOMETRY CHANGES AFTER UNLEARNING:")
            for _, row in sig_changes.head(5).iterrows():
                print(f"   - {row['experiment']}/{row['feature']}: Cohen's d = {row['cohens_d']:.2f}")
    
    print("\n" + "#"*70)
    print("# ANALYSIS COMPLETE")
    print(f"# Results saved to: {output_dir}")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
