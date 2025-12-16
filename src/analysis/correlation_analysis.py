"""
Correlation Analysis with Direction
Run after experiments to see: "more X → more/less Y"
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from pathlib import Path


def analyze_correlations(output_dir: str):
    """
    Analyze correlations between geometric features and outcomes
    Shows DIRECTION: positive = "more X → more Y", negative = "more X → less Y"
    """
    output_dir = Path(output_dir)
    
    # Load data
    base_features = pd.read_csv(output_dir / "base_geometric_features.csv")
    unlearned_features = pd.read_csv(output_dir / "unlearned_geometric_features.csv")
    retention = pd.read_csv(output_dir / "direct_retention.csv")
    attack = pd.read_csv(output_dir / "attack_results.csv")
    
    # Merge
    base_data = base_features.merge(retention, on='query_id').merge(attack[['query_id', 'caf']], on='query_id')
    base_data['unlearning_success'] = 1 - base_data['retention']
    
    unlearned_data = unlearned_features.merge(retention, on='query_id').merge(attack[['query_id', 'caf']], on='query_id')
    unlearned_data['unlearning_success'] = 1 - unlearned_data['retention']
    
    # Feature columns
    feature_cols = [c for c in base_features.columns 
                    if any(x in c for x in ['density', 'separability', 'centrality', 
                                            'isolation', 'compactness', 'consistency'])]
    
    results = []
    
    # Analyze 4 combinations
    analyses = [
        ("Base Geometry", "Unlearning Success", base_data, feature_cols, 'unlearning_success'),
        ("Base Geometry", "Extraction (CAF)", base_data, feature_cols, 'caf'),
        ("Unlearned Geometry", "Unlearning Success", unlearned_data, feature_cols, 'unlearning_success'),
        ("Unlearned Geometry", "Extraction (CAF)", unlearned_data, feature_cols, 'caf'),
    ]
    
    print("=" * 80)
    print("CORRELATION ANALYSIS WITH DIRECTION")
    print("=" * 80)
    
    for geometry_source, target_name, data, features, target_col in analyses:
        print(f"\n{'='*60}")
        print(f"{geometry_source} → {target_name}")
        print(f"{'='*60}")
        print(f"{'Feature':<35} {'Pearson r':>10} {'p-value':>10} {'Direction':<30}")
        print("-" * 85)
        
        for feat in sorted(features):
            x = data[feat].values
            y = data[target_col].values
            
            # Remove NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 3:
                continue
            
            r, p = pearsonr(x_clean, y_clean)
            
            # Interpret direction
            if p < 0.05:  # Significant
                if r > 0.3:
                    direction = f"↑ more {feat.split('_')[0]} → MORE {target_name}"
                elif r < -0.3:
                    direction = f"↓ more {feat.split('_')[0]} → LESS {target_name}"
                elif r > 0:
                    direction = f"↗ weak positive"
                else:
                    direction = f"↘ weak negative"
                sig = "*"
            else:
                direction = "— not significant"
                sig = ""
            
            print(f"{feat:<35} {r:>10.3f} {p:>10.4f} {direction}")
            
            results.append({
                'geometry_source': geometry_source,
                'target': target_name,
                'feature': feat,
                'pearson_r': r,
                'p_value': p,
                'significant': p < 0.05,
                'direction': 'positive' if r > 0 else 'negative',
                'strength': abs(r)
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "correlation_analysis.csv", index=False)
    print(f"\n\nSaved to {output_dir / 'correlation_analysis.csv'}")
    
    # Summary: Top significant correlations
    print("\n" + "=" * 80)
    print("TOP SIGNIFICANT CORRELATIONS (|r| > 0.3, p < 0.05)")
    print("=" * 80)
    
    sig_results = results_df[(results_df['significant']) & (results_df['strength'] > 0.3)]
    sig_results = sig_results.sort_values('strength', ascending=False)
    
    for _, row in sig_results.head(20).iterrows():
        direction_word = "MORE" if row['direction'] == 'positive' else "LESS"
        print(f"• {row['geometry_source']} | {row['feature']}")
        print(f"  → {direction_word} {row['target']} (r={row['pearson_r']:.3f}, p={row['p_value']:.4f})")
        print()
    
    return results_df


def create_interpretation_summary(output_dir: str):
    """
    Create human-readable summary of findings
    """
    output_dir = Path(output_dir)
    corr_df = pd.read_csv(output_dir / "correlation_analysis.csv")
    
    summary_lines = []
    summary_lines.append("# Correlation Interpretation Summary\n")
    
    # Group by target
    for target in ['Unlearning Success', 'Extraction (CAF)']:
        summary_lines.append(f"\n## Predicting {target}\n")
        
        target_df = corr_df[corr_df['target'] == target]
        sig_df = target_df[(target_df['significant']) & (target_df['strength'] > 0.2)]
        sig_df = sig_df.sort_values('strength', ascending=False)
        
        if len(sig_df) == 0:
            summary_lines.append("No significant correlations found.\n")
            continue
        
        for _, row in sig_df.head(10).iterrows():
            feat_name = row['feature'].replace('_mean', '').replace('_std', ' (std)').replace('_max', ' (max)').replace('_min', ' (min)')
            
            if row['direction'] == 'positive':
                summary_lines.append(f"- **Higher {feat_name}** → **Higher {target}** (r={row['pearson_r']:.3f})")
            else:
                summary_lines.append(f"- **Higher {feat_name}** → **Lower {target}** (r={row['pearson_r']:.3f})")
    
    summary_text = "\n".join(summary_lines)
    
    with open(output_dir / "correlation_summary.md", 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\nSaved to {output_dir / 'correlation_summary.md'}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python correlation_analysis.py <output_dir>")
        print("Example: python correlation_analysis.py ./outputs_tofu_1_grad_diff")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    analyze_correlations(output_dir)
    create_interpretation_summary(output_dir)