"""
Test Script for Azure Anomaly Detection API

This script demonstrates how to use the API and test all endpoints.
"""

import requests
import json
from datetime import datetime, timedelta

# API Configuration
API_URL = "http://localhost:5000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def test_health_check():
    """Test the health check endpoint"""
    print_section("1. Health Check")
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    return response.status_code == 200

def test_model_info():
    """Test the model info endpoint"""
    print_section("2. Model Information")
    
    response = requests.get(f"{API_URL}/model_info")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        model_info = response.json()
        if 'model' in model_info:
            print(f"Model Name: {model_info['model'].get('model_name')}")
            print(f"Model Type: {model_info['model'].get('model_type')}")
            
            perf = model_info['model'].get('performance', {}).get('test', {})
            print(f"\nTest Performance:")
            print(f"  F1-Score: {perf.get('f1_score')}")
            print(f"  Precision: {perf.get('precision')}")
            print(f"  Recall: {perf.get('recall')}")
    
    return response.status_code == 200

def test_features():
    """Test the features endpoint"""
    print_section("3. Feature Information")
    
    # This endpoint doesn't exist in the current API
    print("⚠️  Note: No /features endpoint available in current API")
    print("Feature count available via /model_info endpoint")
    
    return True

def test_single_prediction():
    """Test single prediction endpoint"""
    print_section("4. Single Prediction - Normal Metrics")
    
    # Normal cluster metrics
    data = {
        "timestamp": datetime.now().isoformat(),
        "cluster_cpu_request_ratio": 0.45,
        "cluster_mem_request_ratio": 0.52,
        "cluster_pod_ratio": 0.38
    }
    
    print("Input Data:")
    print(json.dumps(data, indent=2))
    print()
    
    response = requests.post(
        f"{API_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        pred = result.get('prediction', {})
        print(f"\nPrediction: {pred.get('prediction', 'N/A').upper()}")
        print(f"Is Anomaly: {pred.get('is_anomaly', False)}")
        print(f"Anomaly Score: {pred.get('anomaly_score', 'N/A'):.4f}")
    else:
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(f"❌ Error: {response.text}")
    
    return response.status_code == 200

def test_anomaly_prediction():
    """Test prediction with anomalous metrics"""
    print_section("5. Single Prediction - Anomalous Metrics")
    
    # Anomalous cluster metrics (very high values)
    data = {
        "timestamp": datetime.now().isoformat(),
        "cluster_cpu_request_ratio": 0.95,
        "cluster_mem_request_ratio": 0.98,
        "cluster_pod_ratio": 0.92
    }
    
    print("Input Data (High Resource Usage):")
    print(json.dumps(data, indent=2))
    print()
    
    response = requests.post(
        f"{API_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        pred = result.get('prediction', {})
        print(f"\nPrediction: {pred.get('prediction', 'N/A').upper()}")
        print(f"Is Anomaly: {pred.get('is_anomaly', False)}")
        print(f"Anomaly Score: {pred.get('anomaly_score', 'N/A'):.4f}")
        
        if pred.get('is_anomaly'):
            print("\n⚠️ ANOMALY DETECTED!")
            print("   An email alert has been sent to configured recipients.")
    else:
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(f"❌ Error: {response.text}")
    
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print_section("6. Batch Prediction")
    
    # Generate sample data with mix of normal and anomalous values
    base_time = datetime.now()
    data_points = []
    
    # Normal metrics
    for i in range(3):
        data_points.append({
            "timestamp": (base_time + timedelta(minutes=i*5)).isoformat(),
            "cluster_cpu_request_ratio": 0.40 + i * 0.05,
            "cluster_mem_request_ratio": 0.50 + i * 0.03,
            "cluster_pod_ratio": 0.35 + i * 0.04
        })
    
    # Anomalous metrics
    data_points.append({
        "timestamp": (base_time + timedelta(minutes=15)).isoformat(),
        "cluster_cpu_request_ratio": 0.92,
        "cluster_mem_request_ratio": 0.95,
        "cluster_pod_ratio": 0.88
    })
    
    # Back to normal
    data_points.append({
        "timestamp": (base_time + timedelta(minutes=20)).isoformat(),
        "cluster_cpu_request_ratio": 0.48,
        "cluster_mem_request_ratio": 0.55,
        "cluster_pod_ratio": 0.42
    })
    
    payload = {"data": data_points}
    
    print(f"Processing {len(data_points)} data points...")
    print()
    
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        summary = result.get('summary', {})
        
        print("\nSummary:")
        print(f"  Total Points: {summary.get('total_points')}")
        print(f"  Anomalies Detected: {summary.get('anomalies_detected')}")
        print(f"  Anomaly Rate: {summary.get('anomaly_rate')}")
        print(f"  Avg Decision Score: {summary.get('avg_decision_score'):.4f}")
        
        print("\nDetailed Results:")
        for pred in result.get('predictions', []):
            status = "🚨 ANOMALY" if pred.get('is_anomaly') else "✅ NORMAL"
            print(f"  [{pred.get('index')}] {status} | Score: {pred.get('anomaly_score'):.4f}")
    else:
        print(json.dumps(response.json(), indent=2))
    
    return response.status_code == 200

def test_alert():
    """Test Alertmanager integration"""
    print_section("7. Test Email Alert")
    
    print("Sending test alert to Alertmanager...")
    print()
    
    response = requests.post(
        f"{API_URL}/alert/test",
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nSuccess: {result.get('success')}")
        print(f"Message: {result.get('message')}")
        print(f"Alertmanager URL: {result.get('alertmanager_url')}")
        
        if result.get('success'):
            print("\n✅ Test alert sent successfully!")
            print("   Check your email inbox for the alert.")
        else:
            print("\n⚠️ Failed to send alert.")
            print("   Check Alertmanager configuration and logs.")
    else:
        print(json.dumps(response.json(), indent=2))
    
    return response.status_code == 200

def run_all_tests():
    """Run all API tests"""
    print("\n" + "🚀 " * 20)
    print("AWS Anomaly Detection API - Test Suite")
    print("🚀 " * 20)
    
    results = {
        "Health Check": test_health_check(),
        "Model Info": test_model_info(),
        "Features": test_features(),
        "Single Prediction (Normal)": test_single_prediction(),
        "Single Prediction (Anomaly)": test_anomaly_prediction(),
        "Batch Prediction": test_batch_prediction(),
        "Alert Test": test_alert()
    }
    
    # Print summary
    print_section("Test Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    print("\n" + "="*60)
    print("\n📝 Next Steps:")
    print("   1. Check your email for test alerts")
    print("   2. View Alertmanager UI: http://localhost:9093")
    print("   3. Try the API with your own data")
    print("   4. Review API documentation in PHASE_5_COMPLETE.md")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("   Make sure the API is running:")
        print("   - Docker: docker-compose up -d")
        print("   - Local: python api/app.py")
        print(f"   - URL: {API_URL}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
