"""
Data Loading Module

Utilities for loading and validating AWS Prometheus cluster metrics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for AWS cluster Prometheus metrics
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def load_prometheus_data(self, metric_name: str) -> pd.DataFrame:
        """
        Load a single Prometheus metric from JSON file
        
        Args:
            metric_name: Name of the metric (e.g., 'cluster_cpu_request_ratio')
            
        Returns:
            DataFrame with timestamp index and metric values
        """
        file_path = self.data_dir / f"{metric_name}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Metric file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract time series data
            if 'data' in data and 'result' in data['data']:
                results = data['data']['result']
                
                if not results:
                    raise ValueError(f"No data found in {metric_name}")
                
                # Combine all time series (usually just one for aggregate metrics)
                all_values = []
                for result in results:
                    values = result.get('values', [])
                    all_values.extend(values)
                
                # Create DataFrame
                df = pd.DataFrame(all_values, columns=['timestamp', metric_name])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df[metric_name] = pd.to_numeric(df[metric_name], errors='coerce')
                df = df.set_index('timestamp').sort_index()
                
                logger.info(f"✅ Loaded {metric_name}: {len(df)} samples")
                return df
            
            else:
                raise ValueError(f"Invalid data structure in {file_path}")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
    
    def load_all_metrics(self, 
                         metric_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load and merge all cluster metrics
        
        Args:
            metric_names: List of metric names. If None, loads default metrics
            
        Returns:
            DataFrame with all metrics aligned by timestamp
        """
        if metric_names is None:
            metric_names = [
                'cluster_cpu_request_ratio',
                'cluster_mem_request_ratio',
                'cluster_pod_ratio'
            ]
        
        logger.info(f"Loading {len(metric_names)} metrics...")
        
        dfs = []
        for metric in metric_names:
            try:
                df = self.load_prometheus_data(metric)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load {metric}: {e}")
                raise
        
        # Merge on timestamp
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = merged_df.join(df, how='outer')
        
        # Forward fill missing values (common in time series)
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"✅ Merged dataset shape: {merged_df.shape}")
        logger.info(f"   Date range: {merged_df.index.min()} to {merged_df.index.max()}")
        
        return merged_df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate loaded data and return statistics
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'date_range': (str(df.index.min()), str(df.index.max())),
            'num_samples': len(df),
            'duplicates': df.duplicated().sum(),
            'statistics': df.describe().to_dict()
        }
        
        # Check for issues
        issues = []
        
        if df.isnull().any().any():
            issues.append(f"Found missing values: {df.isnull().sum().sum()}")
        
        if df.duplicated().any():
            issues.append(f"Found duplicate rows: {df.duplicated().sum()}")
        
        if (df < 0).any().any():
            issues.append("Found negative values (unusual for ratio metrics)")
        
        if (df > 1).any().any():
            issues.append("Found values > 1 (unusual for ratio metrics)")
        
        validation['issues'] = issues
        validation['valid'] = len(issues) == 0
        
        return validation


def load_data(data_dir: str = "data", 
             metric_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convenience function to load all metrics
    
    Args:
        data_dir: Directory containing data files
        metric_names: List of metric names (optional)
        
    Returns:
        DataFrame with all metrics
    """
    loader = DataLoader(data_dir)
    return loader.load_all_metrics(metric_names)


def normalize_field_names(data: Dict) -> Dict:
    """
    Normalize field names in request data
    
    Handles variations like:
    - cpu_ratio vs cluster_cpu_request_ratio
    - memory_ratio vs cluster_mem_request_ratio
    - pod_ratio vs cluster_pod_ratio
    
    Args:
        data: Dictionary with metric data
        
    Returns:
        Dictionary with normalized field names
    """
    field_mapping = {
        # CPU variants
        'cpu_ratio': 'cluster_cpu_request_ratio',
        'cpu': 'cluster_cpu_request_ratio',
        'cluster_cpu_ratio': 'cluster_cpu_request_ratio',
        
        # Memory variants
        'mem_ratio': 'cluster_mem_request_ratio',
        'memory_ratio': 'cluster_mem_request_ratio',
        'mem': 'cluster_mem_request_ratio',
        'memory': 'cluster_mem_request_ratio',
        'cluster_memory_ratio': 'cluster_mem_request_ratio',
        
        # Pod variants
        'pod_ratio': 'cluster_pod_ratio',
        'pod': 'cluster_pod_ratio',
        'pods': 'cluster_pod_ratio',
        'cluster_pods_ratio': 'cluster_pod_ratio'
    }
    
    normalized = {}
    for key, value in data.items():
        # Check if key needs normalization
        normalized_key = field_mapping.get(key, key)
        normalized[normalized_key] = value
    
    return normalized


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Data Loading Module")
    print("Usage: from src.data_loader import DataLoader, load_data")
    print("\nExample:")
    print("  loader = DataLoader('data')")
    print("  df = loader.load_all_metrics()")
    print("  validation = loader.validate_data(df)")
