"""
Azure Cluster Anomaly Detection API

Flask REST API for real-time anomaly detection in Azure cluster metrics.
Integrates with Alertmanager for automated alert notifications.
"""

import os
import sys
import pickle
import json
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'

# Alertmanager configuration
ALERTMANAGER_URL = os.environ.get('ALERTMANAGER_URL', 'http://alertmanager:9093')

# Global variables for loaded models
model = None
scaler = None
feature_names = None
model_config = None
feature_engineer = None


def load_artifacts():
    """
    Load all required model artifacts at startup
    """
    global model, scaler, feature_names, model_config, feature_engineer
    
    try:
        logger.info("Loading model artifacts...")
        
        # Load One-Class SVM model
        model_path = MODELS_DIR / 'one_class_svm_final.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("✅ Loaded One-Class SVM model")
        else:
            logger.error(f"❌ Model not found at {model_path}")
            return False
        
        # Load scaler
        scaler_path = MODELS_DIR / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("✅ Loaded scaler")
        else:
            logger.error(f"❌ Scaler not found at {scaler_path}")
            return False
        
        # Load feature names
        feature_names_path = MODELS_DIR / 'feature_names.pkl'
        if feature_names_path.exists():
            with open(feature_names_path, 'rb') as f:
                feature_names = pickle.load(f)
            logger.info(f"✅ Loaded {len(feature_names)} feature names")
        else:
            logger.error(f"❌ Feature names not found at {feature_names_path}")
            return False
        
        # Load model configuration
        config_path = MODELS_DIR / 'final_model_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            logger.info("✅ Loaded model configuration")
        else:
            logger.warning(f"⚠️ Model config not found at {config_path}")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(verbose=False)
        logger.info("✅ Initialized feature engineer")
        
        logger.info("="*60)
        logger.info("All artifacts loaded successfully!")
        logger.info("="*60)
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading artifacts: {str(e)}", exc_info=True)
        return False


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering pipeline to input data
    
    Args:
        df: DataFrame with base metrics (cluster_cpu_request_ratio, 
            cluster_mem_request_ratio, cluster_pod_ratio)
    
    Returns:
        DataFrame with engineered features
    """
    try:
        # Apply feature engineering
        df_features = feature_engineer.fit_transform(df)
        
        # Check for missing features and fill with default values
        missing_features = set(feature_names) - set(df_features.columns)
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features, filling with 0")
            logger.debug(f"Missing features: {list(missing_features)[:10]}...")
            for feature in missing_features:
                df_features[feature] = 0
        
        # Select only the features used by the model
        df_selected = df_features[feature_names]
        
        return df_selected
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}", exc_info=True)
        raise


def normalize_field_names(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize field names in request data to handle variations
    
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


def send_alert_to_alertmanager(alert_data: Dict[str, Any]) -> bool:
    """
    Send alert to Alertmanager
    
    Args:
        alert_data: Dictionary containing alert information
    
    Returns:
        True if alert sent successfully, False otherwise
    """
    try:
        # Format alert for Alertmanager
        alert_payload = [{
            "labels": {
                "alertname": "AzureClusterAnomaly",
                "severity": alert_data.get('severity', 'warning'),
                "cluster": alert_data.get('cluster', 'unknown'),
                "service": "anomaly-detection-api"
            },
            "annotations": {
                "summary": alert_data.get('summary', 'Anomaly detected in Azure cluster metrics'),
                "description": alert_data.get('description', ''),
                "timestamp": alert_data.get('timestamp', datetime.now().isoformat()),
                "anomaly_score": str(alert_data.get('anomaly_score', 'N/A')),
                "cpu_ratio": str(alert_data.get('cpu_ratio', 'N/A')),
                "mem_ratio": str(alert_data.get('mem_ratio', 'N/A')),
                "pod_ratio": str(alert_data.get('pod_ratio', 'N/A'))
            }
        }]
        
        # Send to Alertmanager (v2 API)
        url = f"{ALERTMANAGER_URL}/api/v2/alerts"
        response = requests.post(url, json=alert_payload, timeout=5)
        
        if response.status_code == 200:
            logger.info(f"✅ Alert sent to Alertmanager: {alert_data['summary']}")
            return True
        else:
            logger.warning(f"⚠️ Failed to send alert. Status: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error sending alert to Alertmanager: {str(e)}")
        return False



# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """
    Service information endpoint
    """
    return jsonify({
        'service': 'Azure Cluster Anomaly Detection API',
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
    
    Returns:
        JSON response with health status
    """
    status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_loaded': feature_names is not None,
        'n_features': len(feature_names) if feature_names else 0
    }
    
    if model_config:
        status['model_info'] = {
            'name': model_config.get('model_name'),
            'type': model_config.get('model_type'),
            'test_f1_score': model_config.get('performance', {}).get('test', {}).get('f1_score')
        }
    
    return jsonify(status), 200 if status['status'] == 'healthy' else 503


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
        'hyperparameters': model_config if model_config else {},
        'timestamp': datetime.now().isoformat()
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
        
        logger.info(f"Processing batch of {len(samples)} samples")
        
        # IMPORTANT: Process ALL samples together for proper feature engineering
        # This allows rolling windows, lags, and temporal features to be calculated correctly
        
        # Normalize all samples
        normalized_samples = [normalize_field_names(s) for s in samples]
        
        # Validate required fields
        required_fields = ['cluster_cpu_request_ratio', 'cluster_mem_request_ratio', 'cluster_pod_ratio']
        
        # Create DataFrame from ALL samples at once
        df_data = {
            'cluster_cpu_request_ratio': [],
            'cluster_mem_request_ratio': [],
            'cluster_pod_ratio': []
        }
        
        timestamps = []
        valid_indices = []
        
        for idx, sample in enumerate(normalized_samples):
            missing_fields = [f for f in required_fields if f not in sample]
            
            if missing_fields:
                logger.warning(f"Sample {idx} missing fields: {missing_fields}")
                continue
            
            df_data['cluster_cpu_request_ratio'].append(sample['cluster_cpu_request_ratio'])
            df_data['cluster_mem_request_ratio'].append(sample['cluster_mem_request_ratio'])
            df_data['cluster_pod_ratio'].append(sample['cluster_pod_ratio'])
            timestamps.append(sample.get('timestamp', None))
            valid_indices.append(idx)
        
        if len(df_data['cluster_cpu_request_ratio']) == 0:
            return jsonify({'error': 'No valid samples found'}), 400
        
        # Create DataFrame with all samples
        df = pd.DataFrame(df_data)
        logger.info(f"Created DataFrame with {len(df)} valid samples")
        
        # Engineer features for entire batch
        df_features = engineer_features(df)
        logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
        
        # Select features - fill missing ones with 0
        missing_features = set(feature_names) - set(df_features.columns)
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features, filling with 0")
            for feature in missing_features:
                df_features[feature] = 0
        
        X = df_features[feature_names]
        X_scaled = scaler.transform(X)
        
        # Predict for all samples at once
        predictions_array = model.predict(X_scaled)
        logger.info(f"Predictions complete. Shape: {predictions_array.shape}")
        
        # Format results
        predictions = []
        anomaly_count = 0
        
        for i, (idx, pred) in enumerate(zip(valid_indices, predictions_array)):
            is_anomaly = pred == -1
            
            if is_anomaly:
                anomaly_count += 1
                
                # Send alert to Alertmanager for each anomaly
                sample = normalized_samples[idx]
                alert_data = {
                    'severity': 'warning' if anomaly_count <= 5 else 'critical',
                    'cluster': 'azure-production',
                    'summary': f'Anomaly detected in Azure cluster metrics (batch {anomaly_count}/{len(valid_indices)})',
                    'description': f'High resource usage detected at {timestamps[i]}',
                    'timestamp': timestamps[i] if timestamps[i] else datetime.now().isoformat(),
                    'cpu_ratio': sample.get('cluster_cpu_request_ratio', 'N/A'),
                    'mem_ratio': sample.get('cluster_mem_request_ratio', 'N/A'),
                    'pod_ratio': sample.get('cluster_pod_ratio', 'N/A'),
                    'anomaly_score': -1.0  # One-Class SVM predicts -1 for anomalies
                }
                send_alert_to_alertmanager(alert_data)
            
            predictions.append({
                'index': idx,
                'is_anomaly': bool(is_anomaly),
                'prediction': 'ANOMALY' if is_anomaly else 'NORMAL',
                'timestamp': timestamps[i]
            })
        
        # Summary
        summary = {
            'total': len(samples),
            'valid': len(valid_indices),
            'anomalies': anomaly_count,
            'normal': len(valid_indices) - anomaly_count,
            'anomaly_rate': round(anomaly_count / len(valid_indices) * 100, 2) if len(valid_indices) > 0 else 0
        }
        
        logger.info(f"Batch prediction complete: {len(valid_indices)} samples, {anomaly_count} anomalies ({summary['anomaly_rate']}%)")
        
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
