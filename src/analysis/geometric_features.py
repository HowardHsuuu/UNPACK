
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr
import warnings


@dataclass
class GeometricFeatures:
    local_density: np.ndarray          # Shape: [num_queries]
    separability: np.ndarray           # Shape: [num_queries]
    centrality: np.ndarray             # Shape: [num_queries]
    isolation: np.ndarray              # Shape: [num_queries]
    cluster_compactness: np.ndarray    # Shape: [num_queries]
    
    cross_layer_consistency: np.ndarray  # Shape: [num_queries]
    
    query_ids: List[str]
    
    layer: int
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            'query_id': self.query_ids,
            'local_density': self.local_density,
            'separability': self.separability,
            'centrality': self.centrality,
            'isolation': self.isolation,
            'cluster_compactness': self.cluster_compactness,
            'cross_layer_consistency': self.cross_layer_consistency,
            'layer': [self.layer] * len(self.query_ids)
        }


class GeometricFeatureComputer:
    def __init__(
        self,
        k_neighbors: int = 10,
        use_pca: bool = True,
        pca_components: int = 50,
        normalize: bool = True
    ):
        self.k_neighbors = k_neighbors
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.normalize = normalize
        
        self.scaler = StandardScaler() if normalize else None
        self.pca = PCA(n_components=pca_components) if use_pca else None
        
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X_processed = X.copy()
        
        if self.normalize:
            X_processed = self.scaler.fit_transform(X_processed)
            
        if self.use_pca and X_processed.shape[1] > self.pca_components:
            X_processed = self.pca.fit_transform(X_processed)
            
        return X_processed
        
    def compute_local_density(
        self,
        X: np.ndarray,
        k: Optional[int] = None
    ) -> np.ndarray:
        k = k or self.k_neighbors
        k = min(k, len(X) - 1)
        
        nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        
        avg_distances = distances[:, 1:].mean(axis=1)
        
        density = 1.0 / (avg_distances + 1e-8)
        
        return density
    
    def compute_separability(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        unique_labels = np.unique(labels)
        separability = np.zeros(len(X))
        
        for i in range(len(X)):
            point = X[i:i+1]
            same_class_mask = labels == labels[i]
            diff_class_mask = ~same_class_mask
            
            if same_class_mask.sum() > 1:
                same_class_dists = cdist(point, X[same_class_mask], metric='cosine')[0]
                same_class_dists = same_class_dists[same_class_dists > 1e-8]
                intra_dist = same_class_dists.mean() if len(same_class_dists) > 0 else 0
            else:
                intra_dist = 0
                
            if diff_class_mask.sum() > 0:
                diff_class_dists = cdist(point, X[diff_class_mask], metric='cosine')[0]
                inter_dist = diff_class_dists.mean()
            else:
                inter_dist = 1.0
                
            if intra_dist > 0:
                separability[i] = inter_dist / intra_dist
            else:
                separability[i] = inter_dist
                
        return separability
    
    def compute_centrality(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        unique_labels = np.unique(labels)
        centrality = np.zeros(len(X))
        
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[label] = X[mask].mean(axis=0)
            
        for i in range(len(X)):
            centroid = centroids[labels[i]]
            dist = cdist(X[i:i+1], centroid.reshape(1, -1), metric='cosine')[0, 0]
            centrality[i] = dist
            
        return centrality
    
    def compute_isolation(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        isolation = np.zeros(len(X))
        
        for i in range(len(X)):
            diff_class_mask = labels != labels[i]
            
            if diff_class_mask.sum() > 0:
                diff_class_dists = cdist(X[i:i+1], X[diff_class_mask], metric='cosine')[0]
                isolation[i] = diff_class_dists.min()
            else:
                isolation[i] = 1.0
                
        return isolation
    
    def compute_cluster_compactness(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        compactness = np.zeros(len(X))
        
        for i in range(len(X)):
            same_class_mask = labels == labels[i]
            same_class_points = X[same_class_mask]
            
            if len(same_class_points) > 1:
                pairwise_dists = pdist(same_class_points, metric='cosine')
                compactness[i] = np.var(pairwise_dists) if len(pairwise_dists) > 0 else 0
            else:
                compactness[i] = 0
                
        return compactness
    
    def compute_cross_layer_consistency(
        self,
        activations_by_layer: Dict[int, np.ndarray],
        query_idx: int
    ) -> float:
        layers = sorted(activations_by_layer.keys())
        if len(layers) < 2:
            return 1.0
            
        correlations = []
        for i, l1 in enumerate(layers[:-1]):
            for l2 in layers[i+1:]:
                act1 = activations_by_layer[l1][query_idx]
                act2 = activations_by_layer[l2][query_idx]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, _ = pearsonr(act1, act2)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                        
        return np.mean(correlations) if correlations else 0.0
    
    def compute_all_features(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        query_ids: List[str],
        layer: int,
        activations_by_layer: Optional[Dict[int, np.ndarray]] = None
    ) -> GeometricFeatures:
        X_processed = self._preprocess(X)
        
        local_density = self.compute_local_density(X_processed)
        separability = self.compute_separability(X_processed, labels)
        centrality = self.compute_centrality(X_processed, labels)
        isolation = self.compute_isolation(X_processed, labels)
        cluster_compactness = self.compute_cluster_compactness(X_processed, labels)
        
        if activations_by_layer is not None and len(activations_by_layer) > 1:
            cross_layer_consistency = np.array([
                self.compute_cross_layer_consistency(activations_by_layer, i)
                for i in range(len(X))
            ])
        else:
            cross_layer_consistency = np.ones(len(X))
            
        return GeometricFeatures(
            local_density=local_density,
            separability=separability,
            centrality=centrality,
            isolation=isolation,
            cluster_compactness=cluster_compactness,
            cross_layer_consistency=cross_layer_consistency,
            query_ids=query_ids,
            layer=layer
        )


def compute_features_all_layers(
    activations_by_layer: Dict[int, np.ndarray],
    labels: np.ndarray,
    query_ids: List[str],
    **kwargs
) -> Dict[int, GeometricFeatures]:
    computer = GeometricFeatureComputer(**kwargs)
    features_by_layer = {}
    
    for layer, X in activations_by_layer.items():
        features = computer.compute_all_features(
            X=X,
            labels=labels,
            query_ids=query_ids,
            layer=layer,
            activations_by_layer=activations_by_layer
        )
        features_by_layer[layer] = features
        
    return features_by_layer


def aggregate_features_across_layers(
    features_by_layer: Dict[int, GeometricFeatures]
) -> Dict[str, np.ndarray]:
    feature_names = ['local_density', 'separability', 'centrality', 
                     'isolation', 'cluster_compactness', 'cross_layer_consistency']
    
    aggregated = {}
    
    for feat_name in feature_names:
        values = np.stack([
            getattr(features_by_layer[l], feat_name)
            for l in sorted(features_by_layer.keys())
        ])  # Shape: [num_layers, num_queries]
        
        aggregated[f'{feat_name}_mean'] = values.mean(axis=0)
        aggregated[f'{feat_name}_std'] = values.std(axis=0)
        aggregated[f'{feat_name}_max'] = values.max(axis=0)
        aggregated[f'{feat_name}_min'] = values.min(axis=0)
        
    first_layer = list(features_by_layer.keys())[0]
    aggregated['query_id'] = features_by_layer[first_layer].query_ids
    
    return aggregated
