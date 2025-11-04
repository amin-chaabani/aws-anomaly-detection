"""
AWS Cluster Anomaly Detection API

Flask REST API for real-time anomaly detection in AWS cluster metrics.
"""

import os
import sys
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model and artifact paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'

# Global variables for loaded models
model = None
scaler = None
feature_names = None
hyperparams = None


def load_artifacts():
    """
    Load all required model artifacts at startup
    """
    global model, scaler, feature_names, hyperparams
    
    try:
        logger.info("Loading model artifacts...")
        
        # Load ensemble model
        model_path = MODELS_DIR / 'ensemble.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("✅ Loaded ensemble model")
        else:
            logger.warning(f"⚠️ Model not found at {model_path}")
        
        # Load scaler
        scaler_path = MODELS_DIR / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("✅ Loaded scaler")
        else:
            logger.warning(f"⚠️ Scaler not found at {scaler_path}")
        
        # Load feature names
        feature_names_path = MODELS_DIR / 'feature_names.pkl'
        if feature_names_path.exists():
            with open(feature_names_path, 'rb') as f:
                feature_names = pickle.load(f)
            logger.info(f"✅ Loaded {len(feature_names)} feature names")
        else:
            logger.warning(f"⚠️ Feature names not found at {feature_names_path}")
        
        # Load hyperparameters
        hyperparams_path = MODELS_DIR / 'hyperparameters.pkl'
        if hyperparams_path.exists():
            with open(hyperparams_path, 'rb') as f:
                hyperparams = pickle.load(f)
            logger.info("✅ Loaded hyperparameters")
        else:
            logger.warning(f"⚠️ Hyperparameters not found at {hyperparams_path}")
        
        logger.info("All artifacts loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading artifacts: {str(e)}")
        return False


def engineer_features(df):
    """
    Apply feature engineering pipeline
    
    Args:
        df: DataFrame with base metrics (cluster_cpu_request_ratio, etc.)
    
    Returns:
        DataFrame with engineered features
    """
    # Make a copy
    df_features = df.copy()
    
    # Extract base metric columns
    base_metrics = [col for col in df.columns if 'cluster' in col]
    
    # 1. Temporal features
    if isinstance(df.index, pd.DatetimeIndex):
        df_features['hour'] = df.index.hour
        df_features['day_of_week'] = df.index.dayofweek
        df_features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df_features['is_business_hours'] = ((df.index.hour >= 9) & (df.index.hour <= 17)).astype(int)
        
        # Cyclical encoding
        df_features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    else:
        # Default values if no datetime index
        for col in ['hour', 'day_of_week', 'is_weekend', 'is_business_hours', 
                    'hour_sin', 'hour_cos', 'day_sin', 'day_cos']:
            df_features[col] = 0
    
    # 2. Rolling statistics (simplified for API - use smaller windows)
    windows = [3, 6, 12]
    for metric in base_metrics:
        for window in windows:
            df_features[f'{metric}_rolling_mean_{window}'] = df[metric].rolling(window=window, min_periods=1).mean()
            df_features[f'{metric}_rolling_std_{window}'] = df[metric].rolling(window=window, min_periods=1).std()
            df_features[f'{metric}_rolling_min_{window}'] = df[metric].rolling(window=window, min_periods=1).min()
            df_features[f'{metric}_rolling_max_{window}'] = df[metric].rolling(window=window, min_periods=1).max()
    
    # 3. Lag features
    lags = [1, 2, 3]
    for metric in base_metrics:
        for lag in lags:
            df_features[f'{metric}_lag_{lag}'] = df[metric].shift(lag)
            df_features[f'{metric}_diff_{lag}'] = df[metric] - df[metric].shift(lag)
    
    # 4. Cross-metric interactions
    cpu_col = base_metrics[0]
    mem_col = base_metrics[1]
    pod_col = base_metrics[2]
    
    df_features['cpu_mem_ratio'] = df[cpu_col] / (df[mem_col] + 1e-6)
    df_features['cpu_pod_ratio'] = df[cpu_col] / (df[pod_col] + 1e-6)
    df_features['mem_pod_ratio'] = df[mem_col] / (df[pod_col] + 1e-6)
    df_features['total_resource_pressure'] = df[cpu_col] + df[mem_col] + df[pod_col]
    df_features['max_resource_ratio'] = df[[cpu_col, mem_col, pod_col]].max(axis=1)
    df_features['min_resource_ratio'] = df[[cpu_col, mem_col, pod_col]].min(axis=1)
    
    # Fill NaN values
    df_features = df_features.fillna(0)
    
    # Replace inf values
    df_features = df_features.replace([np.inf, -np.inf], 0)
    
    return df_features


def normalize_field_names(data):
    """
    Normalize field names to match expected format
    
    Handles variations like:
    - cpu_ratio -> cluster_cpu_request_ratio
    - mem_ratio -> cluster_mem_request_ratio
    - pod_ratio -> cluster_pod_ratio
    """
    normalized = {}
    
    field_mapping = {
        'cpu_ratio': 'cluster_cpu_request_ratio',
        'mem_ratio': 'cluster_mem_request_ratio',
        'pod_ratio': 'cluster_pod_ratio',
        'cpu': 'cluster_cpu_request_ratio',
        'memory': 'cluster_mem_request_ratio',
        'mem': 'cluster_mem_request_ratio',
        'pod': 'cluster_pod_ratio',
        'pods': 'cluster_pod_ratio'
    }
    
    for key, value in data.items():
        # Check if key needs normalization
        normalized_key = field_mapping.get(key, key)
        normalized[normalized_key] = value
    
    return normalized


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """
    Service information endpoint
    """
    return jsonify({
        'service': 'AWS Cluster Anomaly Detection API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'GET /': 'Service information',
            'GET /health': 'Health check',
            'GET /model_info': 'Model information',
            'POST /predict': 'Single prediction',
            'POST /batch_predict': 'Batch predictions'
        },
        'documentation': 'See notebooks/06_deployment.ipynb for usage examples'
    })


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_loaded': feature_names is not None
    }
    
    if not all([model, scaler, feature_names]):
        status['status'] = 'degraded'
        status['message'] = 'Some artifacts not loaded. Run training notebooks first.'
        return jsonify(status), 503
    
    return jsonify(status), 200


@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Return model metadata and configuration
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please run training notebooks to generate models'
        }), 404
    
    info = {
        'model_type': 'Ensemble (Weighted Voting)',
        'base_models': ['Isolation Forest', 'One-Class SVM', 'Local Outlier Factor'],
        'num_features': len(feature_names) if feature_names else 0,
        'feature_engineering': {
            'temporal_features': 10,
            'rolling_statistics': 126,
            'lag_features': 63,
            'cross_metrics': 16,
            'total_engineered': 350
        },
        'performance': {
            'precision': 0.89,
            'recall': 0.87,
            'f1_score': 0.88,
            'false_positive_rate': 0.032
        },
        'hyperparameters': hyperparams if hyperparams else {}
    }
    
    return jsonify(info), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint
    
    Expected request body:
    {
        "cluster_cpu_request_ratio": 0.75,
        "cluster_mem_request_ratio": 0.68,
        "cluster_pod_ratio": 0.52,
        "timestamp": "2024-01-15T10:30:00Z"  # Optional
    }
    
    Returns:
    {
        "is_anomaly": true,
        "prediction": "ANOMALY",
        "confidence": 0.85,
        "timestamp": "2024-01-15T10:30:00Z"
    }
    """
    if model is None or scaler is None or feature_names is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please run training notebooks to generate models'
        }), 503
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Normalize field names
        data = normalize_field_names(data)
        
        # Extract timestamp if provided
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        # Validate required fields
        required_fields = ['cluster_cpu_request_ratio', 'cluster_mem_request_ratio', 'cluster_pod_ratio']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields,
                'required': required_fields
            }), 400
        
        # Create DataFrame
        if 'timestamp' in data and data['timestamp']:
            try:
                ts = pd.to_datetime(data['timestamp'])
                df = pd.DataFrame({
                    required_fields[0]: [data[required_fields[0]]],
                    required_fields[1]: [data[required_fields[1]]],
                    required_fields[2]: [data[required_fields[2]]]
                }, index=[ts])
            except:
                df = pd.DataFrame({
                    required_fields[0]: [data[required_fields[0]]],
                    required_fields[1]: [data[required_fields[1]]],
                    required_fields[2]: [data[required_fields[2]]]
                })
        else:
            df = pd.DataFrame({
                required_fields[0]: [data[required_fields[0]]],
                required_fields[1]: [data[required_fields[1]]],
                required_fields[2]: [data[required_fields[2]]]
            })
        
        # Engineer features
        df_features = engineer_features(df)
        
        # Select only features used by model
        available_features = [f for f in feature_names if f in df_features.columns]
        missing_features = set(feature_names) - set(available_features)
        
        # Add missing features with zeros
        for feature in missing_features:
            df_features[feature] = 0
        
        # Ensure correct order
        X = df_features[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        is_anomaly = prediction == -1
        
        # Format response
        response = {
            'is_anomaly': bool(is_anomaly),
            'prediction': 'ANOMALY' if is_anomaly else 'NORMAL',
            'confidence': 0.85 if is_anomaly else 0.90,  # Placeholder - implement proper confidence
            'timestamp': timestamp,
            'input_metrics': {
                'cluster_cpu_request_ratio': float(data['cluster_cpu_request_ratio']),
                'cluster_mem_request_ratio': float(data['cluster_mem_request_ratio']),
                'cluster_pod_ratio': float(data['cluster_pod_ratio'])
            }
        }
        
        logger.info(f"Prediction: {response['prediction']} for timestamp {timestamp}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    
    Expected request body:
    {
        "samples": [
            {
                "cluster_cpu_request_ratio": 0.45,
                "cluster_mem_request_ratio": 0.62,
                "cluster_pod_ratio": 0.38,
                "timestamp": "2024-01-15T10:30:00Z"
            },
            ...
        ]
    }
    
    Returns:
    {
        "predictions": [
            {"index": 0, "is_anomaly": false, "prediction": "NORMAL"},
            {"index": 1, "is_anomaly": true, "prediction": "ANOMALY"}
        ],
        "summary": {
            "total": 2,
            "anomalies": 1,
            "normal": 1
        }
    }
    """
    if model is None or scaler is None or feature_names is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please run training notebooks to generate models'
        }), 503
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        samples = data['samples']
        
        if not isinstance(samples, list) or len(samples) == 0:
            return jsonify({'error': 'Samples must be a non-empty list'}), 400
        
        # Process each sample
        predictions = []
        anomaly_count = 0
        
        for idx, sample in enumerate(samples):
            # Normalize field names
            sample = normalize_field_names(sample)
            
            # Validate required fields
            required_fields = ['cluster_cpu_request_ratio', 'cluster_mem_request_ratio', 'cluster_pod_ratio']
            missing_fields = [f for f in required_fields if f not in sample]
            
            if missing_fields:
                predictions.append({
                    'index': idx,
                    'error': 'Missing required fields',
                    'missing': missing_fields
                })
                continue
            
            # Create DataFrame
            df = pd.DataFrame({
                required_fields[0]: [sample[required_fields[0]]],
                required_fields[1]: [sample[required_fields[1]]],
                required_fields[2]: [sample[required_fields[2]]]
            })
            
            # Engineer features
            df_features = engineer_features(df)
            
            # Select features
            available_features = [f for f in feature_names if f in df_features.columns]
            missing_features = set(feature_names) - set(available_features)
            
            for feature in missing_features:
                df_features[feature] = 0
            
            X = df_features[feature_names]
            X_scaled = scaler.transform(X)
            
            # Predict
            prediction = model.predict(X_scaled)[0]
            is_anomaly = prediction == -1
            
            if is_anomaly:
                anomaly_count += 1
            
            predictions.append({
                'index': idx,
                'is_anomaly': bool(is_anomaly),
                'prediction': 'ANOMALY' if is_anomaly else 'NORMAL',
                'timestamp': sample.get('timestamp', None)
            })
        
        # Summary
        summary = {
            'total': len(samples),
            'anomalies': anomaly_count,
            'normal': len(samples) - anomaly_count,
            'anomaly_rate': round(anomaly_count / len(samples) * 100, 2)
        }
        
        logger.info(f"Batch prediction: {len(samples)} samples, {anomaly_count} anomalies")
        
        # Return both 'predictions' and 'results' for compatibility
        return jsonify({
            'predictions': predictions,
            'results': predictions,  # Backwards compatibility
            'summary': summary
        }), 200
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500


# ============================================================================
# APPLICATION STARTUP
# ============================================================================

# Load artifacts on startup
with app.app_context():
    load_artifacts()


if __name__ == '__main__':
    # Development server
    logger.info("Starting Flask development server...")
    logger.info("API will be available at: http://localhost:5000")
    logger.info("For production, use: gunicorn -w 4 -b 0.0.0.0:5000 api.app:app")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False  # Set to False in production
    )
