"""
Utility Functions Module

Common utilities for model management, evaluation, and helper functions.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import logging

logger = logging.getLogger(__name__)


def load_model_artifacts(model_dir: str = "models") -> Tuple[Any, Any, List[str], Dict]:
    """
    Load all model artifacts from directory
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Tuple of (model, scaler, feature_names, metadata)
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load model
    model_file = model_path / "best_model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"✅ Loaded model from {model_file}")
    
    # Load scaler
    scaler_file = model_path / "scaler.pkl"
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
    
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"✅ Loaded scaler from {scaler_file}")
    
    # Load feature names
    features_file = model_path / "feature_names.pkl"
    if not features_file.exists():
        raise FileNotFoundError(f"Feature names file not found: {features_file}")
    
    with open(features_file, 'rb') as f:
        feature_names = pickle.load(f)
    logger.info(f"✅ Loaded {len(feature_names)} feature names")
    
    # Load metadata
    metadata_file = model_path / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        logger.info(f"✅ Loaded metadata")
    else:
        logger.warning("Metadata file not found")
    
    return model, scaler, feature_names, metadata


def save_model_artifacts(model: Any,
                        scaler: Any,
                        feature_names: List[str],
                        metadata: Dict,
                        model_dir: str = "models") -> None:
    """
    Save all model artifacts to directory
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_names: List of feature names
        metadata: Model metadata dictionary
        model_dir: Directory to save files
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(model_path / "best_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"✅ Saved model")
    
    # Save scaler
    with open(model_path / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"✅ Saved scaler")
    
    # Save feature names
    with open(model_path / "feature_names.pkl", 'wb') as f:
        pickle.dump(feature_names, f)
    logger.info(f"✅ Saved feature names")
    
    # Save metadata
    with open(model_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata")


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     model_name: str = "Model") -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: True labels (-1 for anomaly, 1 for normal)
        y_pred: Predicted labels
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with all metrics
    """
    # Convert to binary (0 = normal, 1 = anomaly)
    y_true_binary = (y_true == -1).astype(int)
    y_pred_binary = (y_pred == -1).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    # Additional metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    metrics = {
        'model': model_name,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'accuracy': round(accuracy, 4),
        'false_positive_rate': round(fpr, 4),
        'false_negative_rate': round(fnr, 4),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_samples': int(len(y_true)),
        'total_anomalies': int(y_true_binary.sum()),
        'predicted_anomalies': int(y_pred_binary.sum())
    }
    
    return metrics


def print_metrics_summary(metrics: Dict[str, float]) -> None:
    """
    Print formatted metrics summary
    
    Args:
        metrics: Dictionary with metrics from calculate_metrics
    """
    print("\n" + "="*60)
    print(f"Model: {metrics['model']}")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 Score:  {metrics['f1_score']:.2%}")
    print(f"\nFalse Positive Rate: {metrics['false_positive_rate']:.2%}")
    print(f"False Negative Rate: {metrics['false_negative_rate']:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print("="*60 + "\n")


def detect_anomalies(model: Any,
                    X: np.ndarray,
                    threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies using fitted model
    
    Args:
        model: Fitted anomaly detection model
        X: Feature matrix
        threshold: Optional custom threshold for decision function
        
    Returns:
        Tuple of (predictions, scores)
        predictions: -1 for anomaly, 1 for normal
        scores: Anomaly scores (lower = more anomalous)
    """
    # Get predictions
    predictions = model.predict(X)
    
    # Get anomaly scores if available
    scores = None
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
    elif hasattr(model, 'score_samples'):
        scores = model.score_samples(X)
    
    # Apply custom threshold if provided
    if threshold is not None and scores is not None:
        predictions = np.where(scores < threshold, -1, 1)
    
    return predictions, scores


def create_classification_report(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 target_names: Optional[List[str]] = None) -> str:
    """
    Create detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional names for classes
        
    Returns:
        Formatted classification report string
    """
    # Convert to binary
    y_true_binary = (y_true == -1).astype(int)
    y_pred_binary = (y_pred == -1).astype(int)
    
    if target_names is None:
        target_names = ['Normal', 'Anomaly']
    
    report = classification_report(
        y_true_binary, 
        y_pred_binary,
        target_names=target_names,
        digits=4
    )
    
    return report


def split_temporal_data(df: pd.DataFrame,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data temporally (no shuffling)
    
    Args:
        df: DataFrame with temporal index
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


def handle_inf_nan(df: pd.DataFrame, 
                  fill_value: float = 0,
                  verbose: bool = True) -> pd.DataFrame:
    """
    Handle infinite and NaN values in DataFrame
    
    Args:
        df: Input DataFrame
        fill_value: Value to replace inf/nan with
        verbose: Whether to print info
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Count issues
    n_inf = np.isinf(df.values).sum()
    n_nan = df.isnull().sum().sum()
    
    if verbose and (n_inf > 0 or n_nan > 0):
        logger.info(f"Found {n_inf} infinite values and {n_nan} NaN values")
    
    # Replace
    df = df.replace([np.inf, -np.inf], fill_value)
    df = df.fillna(fill_value)
    
    if verbose:
        logger.info(f"✅ Replaced with {fill_value}")
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Utility Functions Module")
    print("Usage: from src.utils import load_model_artifacts, calculate_metrics, ...")
