import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    

def plot_feature_distributions(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    group_col: str = 'dataset',
    save_path: Optional[str] = None
):
    set_style()
    
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        for group in features_df[group_col].unique():
            data = features_df[features_df[group_col] == group][col]
            ax.hist(data, bins=30, alpha=0.5, label=group, density=True)
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'Distribution of {col}')
        
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    

def plot_feature_correlations(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    save_path: Optional[str] = None
):
    set_style()
    
    corr = features_df[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        ax=ax, square=True
    )
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    

def plot_feature_vs_vulnerability(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    vulnerability_col: str = 'caf',
    save_path: Optional[str] = None
):
    set_style()
    
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        ax.scatter(
            features_df[col], features_df[vulnerability_col],
            alpha=0.5, c=features_df['dataset'].astype('category').cat.codes
        )
        
        z = np.polyfit(features_df[col].dropna(), 
                      features_df.loc[features_df[col].notna(), vulnerability_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(features_df[col].min(), features_df[col].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.8)
        
        corr = features_df[col].corr(features_df[vulnerability_col])
        ax.set_xlabel(col)
        ax.set_ylabel(vulnerability_col)
        ax.set_title(f'{col} vs {vulnerability_col}\n(r={corr:.3f})')
        
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_layer_wise_features(
    features_by_layer: Dict[int, pd.DataFrame],
    feature_name: str,
    group_col: str = 'dataset',
    save_path: Optional[str] = None
):
    set_style()
    
    layers = sorted(features_by_layer.keys())
    groups = features_by_layer[layers[0]][group_col].unique()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for group in groups:
        means = []
        stds = []
        for layer in layers:
            data = features_by_layer[layer]
            group_data = data[data[group_col] == group][feature_name]
            means.append(group_data.mean())
            stds.append(group_data.std())
            
        means = np.array(means)
        stds = np.array(stds)
        
        ax.plot(layers, means, 'o-', label=group)
        ax.fill_between(layers, means - stds, means + stds, alpha=0.2)
        
    ax.set_xlabel('Layer')
    ax.set_ylabel(feature_name)
    ax.set_title(f'{feature_name} Across Layers')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_prediction_results(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    model_name: str,
    task_type: str = 'regression',
    save_path: Optional[str] = None
):
    set_style()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if task_type == 'regression':
        ax.scatter(ground_truth, predictions, alpha=0.5)
        
        min_val = min(ground_truth.min(), predictions.min())
        max_val = max(ground_truth.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        
        ss_res = np.sum((ground_truth - predictions) ** 2)
        ss_tot = np.sum((ground_truth - ground_truth.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{model_name}\nR² = {r2:.3f}')
        ax.legend()
        
    else:  # classification
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(ground_truth, predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{model_name} Confusion Matrix')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[str] = None
):
    set_style()
    
    df = importance_df.head(top_n).sort_values('mean')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df['mean'], xerr=df['std'], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (Mean ± Std across models)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_umap_activations(
    activations: np.ndarray,
    labels: np.ndarray,
    layer: int,
    save_path: Optional[str] = None
):
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return
        
    set_style()
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(activations)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            alpha=0.6, label=label, s=30
        )
        
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'UMAP Visualization of Layer {layer} Activations')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def create_summary_report(
    features_df: pd.DataFrame,
    prediction_results: Dict,
    output_dir: str
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_cols = [c for c in features_df.columns 
                   if any(x in c for x in ['density', 'separability', 'centrality',
                                           'isolation', 'compactness', 'consistency'])]
    
    plot_feature_distributions(
        features_df, feature_cols[:6],
        save_path=str(output_dir / 'feature_distributions.png')
    )
    
    plot_feature_correlations(
        features_df, feature_cols,
        save_path=str(output_dir / 'feature_correlations.png')
    )
    
    if 'caf' in features_df.columns:
        plot_feature_vs_vulnerability(
            features_df, feature_cols[:6],
            save_path=str(output_dir / 'feature_vs_vulnerability.png')
        )
        
    if prediction_results and 'regression' in prediction_results:
        for model_name, result in prediction_results['regression'].items():
            plot_prediction_results(
                result.predictions, result.ground_truth,
                model_name, 'regression',
                save_path=str(output_dir / f'prediction_{model_name}.png')
            )
            
    if prediction_results and 'importance' in prediction_results:
        if not prediction_results['importance'].empty:
            plot_feature_importance(
                prediction_results['importance'],
                save_path=str(output_dir / 'feature_importance.png')
            )
            
    print(f"Summary report saved to {output_dir}")
