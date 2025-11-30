import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    roc_auc_score, classification_report
)
import warnings


@dataclass
class PredictionResults:
    model_name: str
    task_type: str  # "regression" or "classification"
    cv_scores: np.ndarray
    test_score: float
    feature_importances: Optional[Dict[str, float]]
    predictions: np.ndarray
    ground_truth: np.ndarray
    

class VulnerabilityPredictor:
    
    REGRESSION_MODELS = {
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
    }
    
    CLASSIFICATION_MODELS = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
    }
    
    def __init__(
        self,
        task_type: str = "regression",
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.models = (
            self.REGRESSION_MODELS if task_type == "regression" 
            else self.CLASSIFICATION_MODELS
        )
        
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if feature_cols is None:
            feature_cols = [
                col for col in features_df.select_dtypes(include=[np.number]).columns
                if col != target_col
            ]
            
        X = features_df[feature_cols].values
        y = features_df[target_col].values
        
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        return X, y, feature_cols
        
    def train_evaluate_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        feature_names: List[str]
    ) -> PredictionResults:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_class = self.models[model_name]
        
        if 'random_forest' in model_name:
            model = model_class(n_estimators=100, random_state=self.random_state)
        elif 'gradient_boosting' in model_name:
            model = model_class(n_estimators=100, random_state=self.random_state)
        elif 'logistic_regression' in model_name:
            model = model_class(random_state=self.random_state, max_iter=1000)
        elif 'ridge' in model_name:
            model = model_class(alpha=1.0)
        else:
            model = model_class()
            
        scoring = 'r2' if self.task_type == "regression" else 'accuracy'
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=self.cv_folds, scoring=scoring
        )
        
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        
        if self.task_type == "regression":
            test_score = r2_score(y_test, predictions)
        else:
            test_score = accuracy_score(y_test, predictions)
            
        feature_importances = None
        if hasattr(model, 'feature_importances_'):
            feature_importances = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            feature_importances = dict(zip(feature_names, np.abs(coef)))
            
        return PredictionResults(
            model_name=model_name,
            task_type=self.task_type,
            cv_scores=cv_scores,
            test_score=test_score,
            feature_importances=feature_importances,
            predictions=predictions,
            ground_truth=y_test
        )
        
    def train_all_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, PredictionResults]:
        results = {}
        
        for model_name in self.models.keys():
            try:
                result = self.train_evaluate_model(X, y, model_name, feature_names)
                results[model_name] = result
                print(f"{model_name}: CV={result.cv_scores.mean():.3f}±{result.cv_scores.std():.3f}, "
                      f"Test={result.test_score:.3f}")
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                
        return results
        
    def get_best_model(
        self,
        results: Dict[str, PredictionResults]
    ) -> Tuple[str, PredictionResults]:
        best_name = max(results.keys(), key=lambda k: results[k].test_score)
        return best_name, results[best_name]


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    try:
        import shap
        
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)
            
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        
        return shap_df
        
    except ImportError:
        warnings.warn("SHAP not installed. Install with: pip install shap")
        return pd.DataFrame()
    except Exception as e:
        warnings.warn(f"Error computing SHAP values: {e}")
        return pd.DataFrame()


def create_feature_importance_summary(
    results: Dict[str, PredictionResults]
) -> pd.DataFrame:
    importance_data = []
    
    for model_name, result in results.items():
        if result.feature_importances:
            for feat, imp in result.feature_importances.items():
                importance_data.append({
                    'model': model_name,
                    'feature': feat,
                    'importance': imp
                })
                
    if not importance_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(importance_data)
    
    summary = df.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
    summary = summary.sort_values('mean', ascending=False)
    
    return summary


def analyze_prediction_errors(
    results: PredictionResults,
    features_df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    errors = results.predictions - results.ground_truth
    
    error_indices = np.argsort(np.abs(errors))[::-1]
    
    analysis_data = []
    for idx in error_indices[:20]:
        row = {
            'prediction': results.predictions[idx],
            'ground_truth': results.ground_truth[idx],
            'error': errors[idx],
            'abs_error': np.abs(errors[idx])
        }
        analysis_data.append(row)
        
    return pd.DataFrame(analysis_data)
