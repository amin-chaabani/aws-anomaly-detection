"""
Feature Engineering Module

Advanced feature generation pipeline for AWS cluster metrics.
Generates 350+ features from 3 base metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats, signal, fft
from typing import List, Dict, Optional


class FeatureEngineer:
    """
    Feature engineering pipeline for time series cluster metrics
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize feature engineer
        
        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self.feature_count = 0
        
    def log(self, message: str):
        """Print message if verbose"""
        if self.verbose:
            print(message)
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime index
        
        Args:
            df: DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with added temporal features
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            self.log("⚠️ No DatetimeIndex found, skipping temporal features")
            return df
        
        df = df.copy()
        
        # Basic temporal features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_business_hours'] = ((df.index.hour >= 9) & (df.index.hour <= 17)).astype(int)
        
        # Cyclical encoding (sine/cosine)
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df.index.day / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df.index.day / 31)
        
        self.feature_count += 12
        self.log(f"✅ Created 12 temporal features")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               base_metrics: List[str],
                               windows: List[int] = [3, 6, 12, 24, 48, 96]) -> pd.DataFrame:
        """
        Create rolling window statistical features
        
        Args:
            df: Input DataFrame
            base_metrics: List of base metric column names
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for metric in base_metrics:
            for window in windows:
                # Mean
                df[f'{metric}_rolling_mean_{window}'] = df[metric].rolling(
                    window=window, min_periods=1).mean()
                
                # Standard deviation
                df[f'{metric}_rolling_std_{window}'] = df[metric].rolling(
                    window=window, min_periods=1).std()
                
                # Min/Max
                df[f'{metric}_rolling_min_{window}'] = df[metric].rolling(
                    window=window, min_periods=1).min()
                df[f'{metric}_rolling_max_{window}'] = df[metric].rolling(
                    window=window, min_periods=1).max()
                
                # Median
                df[f'{metric}_rolling_median_{window}'] = df[metric].rolling(
                    window=window, min_periods=1).median()
                
                # Skewness (require min 3 points or window size, whichever is smaller)
                skew_min_periods = min(3, window)
                df[f'{metric}_rolling_skew_{window}'] = df[metric].rolling(
                    window=window, min_periods=skew_min_periods).skew()
                
                # Kurtosis (require min 4 points or window size, whichever is smaller)
                kurt_min_periods = min(4, window)
                df[f'{metric}_rolling_kurt_{window}'] = df[metric].rolling(
                    window=window, min_periods=kurt_min_periods).kurt()
        
        # Fill NaN values
        rolling_cols = [col for col in df.columns if 'rolling' in col]
        df[rolling_cols] = df[rolling_cols].fillna(method='bfill').fillna(0)
        
        num_features = len(base_metrics) * len(windows) * 7
        self.feature_count += num_features
        self.log(f"✅ Created {num_features} rolling window features")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame,
                           base_metrics: List[str],
                           lags: List[int] = [1, 2, 3, 6, 12, 24, 48]) -> pd.DataFrame:
        """
        Create lag and difference features
        
        Args:
            df: Input DataFrame
            base_metrics: List of base metric column names
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for metric in base_metrics:
            for lag in lags:
                # Lag feature
                df[f'{metric}_lag_{lag}'] = df[metric].shift(lag)
                
                # Difference from lagged value
                df[f'{metric}_diff_{lag}'] = df[metric] - df[metric].shift(lag)
                
                # Percent change from lagged value
                df[f'{metric}_pct_change_{lag}'] = df[metric].pct_change(lag)
        
        # Fill NaN and inf values
        lag_cols = [col for col in df.columns if any(x in col for x in ['_lag_', '_diff_', '_pct_change_'])]
        df[lag_cols] = df[lag_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        num_features = len(base_metrics) * len(lags) * 3
        self.feature_count += num_features
        self.log(f"✅ Created {num_features} lag and difference features")
        
        return df
    
    def create_rate_of_change_features(self, df: pd.DataFrame,
                                      base_metrics: List[str],
                                      windows: List[int] = [3, 6, 12, 24, 48]) -> pd.DataFrame:
        """
        Create rate of change and acceleration features
        
        Args:
            df: Input DataFrame
            base_metrics: List of base metric column names
            windows: List of window sizes
            
        Returns:
            DataFrame with rate of change features
        """
        df = df.copy()
        
        for metric in base_metrics:
            for window in windows:
                # First derivative (rate of change)
                df[f'{metric}_roc_{window}'] = df[metric].diff(window)
                
                # Second derivative (acceleration)
                df[f'{metric}_acceleration_{window}'] = df[f'{metric}_roc_{window}'].diff(1)
                
                # Rolling rate of change
                df[f'{metric}_rolling_roc_{window}'] = df[metric].rolling(
                    window=window, min_periods=1).apply(
                    lambda x: (x.iloc[-1] - x.iloc[0]) / window if len(x) > 1 else 0
                )
        
        # Fill NaN and inf values
        roc_cols = [col for col in df.columns if any(x in col for x in ['_roc_', '_acceleration_'])]
        df[roc_cols] = df[roc_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        num_features = len(base_metrics) * len(windows) * 3
        self.feature_count += num_features
        self.log(f"✅ Created {num_features} rate of change features")
        
        return df
    
    def create_cross_metric_features(self, df: pd.DataFrame,
                                    base_metrics: List[str]) -> pd.DataFrame:
        """
        Create cross-metric interaction features
        
        Args:
            df: Input DataFrame
            base_metrics: List of base metric column names (must have exactly 3)
            
        Returns:
            DataFrame with cross-metric features
        """
        if len(base_metrics) != 3:
            self.log(f"⚠️ Expected 3 base metrics, got {len(base_metrics)}")
            return df
        
        df = df.copy()
        
        cpu_col, mem_col, pod_col = base_metrics
        
        # Ratio features
        df['cpu_mem_ratio'] = df[cpu_col] / (df[mem_col] + 1e-6)
        df['cpu_pod_ratio'] = df[cpu_col] / (df[pod_col] + 1e-6)
        df['mem_pod_ratio'] = df[mem_col] / (df[pod_col] + 1e-6)
        
        # Product features (resource pressure indicators)
        df['cpu_mem_product'] = df[cpu_col] * df[mem_col]
        df['cpu_pod_product'] = df[cpu_col] * df[pod_col]
        df['mem_pod_product'] = df[mem_col] * df[pod_col]
        df['all_metrics_product'] = df[cpu_col] * df[mem_col] * df[pod_col]
        
        # Difference features
        df['cpu_mem_diff'] = df[cpu_col] - df[mem_col]
        df['cpu_pod_diff'] = df[cpu_col] - df[pod_col]
        df['mem_pod_diff'] = df[mem_col] - df[pod_col]
        
        # Aggregate features
        df['total_resource_pressure'] = df[cpu_col] + df[mem_col] + df[pod_col]
        df['max_resource_ratio'] = df[[cpu_col, mem_col, pod_col]].max(axis=1)
        df['min_resource_ratio'] = df[[cpu_col, mem_col, pod_col]].min(axis=1)
        df['resource_range'] = df['max_resource_ratio'] - df['min_resource_ratio']
        df['resource_std'] = df[[cpu_col, mem_col, pod_col]].std(axis=1)
        
        mean_resources = df[[cpu_col, mem_col, pod_col]].mean(axis=1)
        df['resource_cv'] = df['resource_std'] / (mean_resources + 1e-6)
        
        # Replace inf values
        cross_cols = [col for col in df.columns if any(x in col for x in 
                     ['ratio', 'product', 'diff', 'pressure', 'resource_'])]
        df[cross_cols] = df[cross_cols].replace([np.inf, -np.inf], 0)
        
        num_features = 16
        self.feature_count += num_features
        self.log(f"✅ Created {num_features} cross-metric features")
        
        return df
    
    def create_distribution_features(self, df: pd.DataFrame,
                                    base_metrics: List[str],
                                    window: int = 24) -> pd.DataFrame:
        """
        Create statistical distribution features
        
        Args:
            df: Input DataFrame
            base_metrics: List of base metric column names
            window: Window size for rolling statistics
            
        Returns:
            DataFrame with distribution features
        """
        df = df.copy()
        
        for metric in base_metrics:
            # Quantiles
            df[f'{metric}_q25'] = df[metric].rolling(window=window, min_periods=1).quantile(0.25)
            df[f'{metric}_q75'] = df[metric].rolling(window=window, min_periods=1).quantile(0.75)
            df[f'{metric}_iqr'] = df[f'{metric}_q75'] - df[f'{metric}_q25']
            
            # Z-score (standardized value within window)
            rolling_mean = df[metric].rolling(window=window, min_periods=1).mean()
            rolling_std = df[metric].rolling(window=window, min_periods=1).std()
            df[f'{metric}_zscore'] = (df[metric] - rolling_mean) / (rolling_std + 1e-6)
            
            # Distance from median
            rolling_median = df[metric].rolling(window=window, min_periods=1).median()
            df[f'{metric}_dist_from_median'] = np.abs(df[metric] - rolling_median)
            
            # Coefficient of variation
            df[f'{metric}_cv'] = rolling_std / (rolling_mean + 1e-6)
            
            # Is outlier (beyond 2 std devs)
            df[f'{metric}_is_outlier'] = (np.abs(df[f'{metric}_zscore']) > 2).astype(int)
            
            # Percentile rank within window
            df[f'{metric}_percentile'] = df[metric].rolling(window=window, min_periods=1).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 1 else 0.5
            )
            
            # Distance from boundaries
            rolling_min = df[metric].rolling(window=window, min_periods=1).min()
            rolling_max = df[metric].rolling(window=window, min_periods=1).max()
            df[f'{metric}_dist_from_min'] = df[metric] - rolling_min
            df[f'{metric}_dist_from_max'] = rolling_max - df[metric]
        
        # Clean up inf and NaN values
        dist_cols = [col for col in df.columns if any(x in col for x in 
                    ['_q25', '_q75', '_iqr', '_zscore', '_dist_', '_cv', '_is_outlier', '_percentile'])]
        df[dist_cols] = df[dist_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        num_features = len(base_metrics) * 10
        self.feature_count += num_features
        self.log(f"✅ Created {num_features} distribution features")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline
        
        Args:
            df: Input DataFrame with base metrics
            
        Returns:
            DataFrame with all engineered features
        """
        self.log("\n" + "="*60)
        self.log("Starting Feature Engineering Pipeline")
        self.log("="*60)
        
        self.feature_count = 0
        
        # Identify base metrics
        base_metrics = [col for col in df.columns if 'cluster' in col]
        self.log(f"\nBase metrics: {base_metrics}")
        
        # Apply transformations
        df = self.create_temporal_features(df)
        df = self.create_rolling_features(df, base_metrics)
        df = self.create_lag_features(df, base_metrics)
        df = self.create_rate_of_change_features(df, base_metrics)
        df = self.create_cross_metric_features(df, base_metrics)
        df = self.create_distribution_features(df, base_metrics)
        
        self.log("\n" + "="*60)
        self.log(f"Feature Engineering Complete!")
        self.log(f"Total features created: {self.feature_count}")
        self.log(f"Final shape: {df.shape}")
        self.log("="*60 + "\n")
        
        return df


def quick_feature_engineering(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Quick feature engineering function (same as FeatureEngineer.fit_transform)
    
    Args:
        df: Input DataFrame with base metrics
        verbose: Whether to print progress
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer(verbose=verbose)
    return engineer.fit_transform(df)


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("Usage: from src.feature_engineering import FeatureEngineer, quick_feature_engineering")
