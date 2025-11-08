"""
Test script for batch prediction endpoint
Uses real data from JSON files
"""

import json
import requests
from datetime import datetime
from pathlib import Path

API_URL = "http://localhost:5000"

def load_metric_data(filename):
    """Load metric data from JSON file"""
    file_path = Path(__file__).parent / 'data' / filename
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract values from Prometheus format
    values = data['data']['result'][0]['values']
    
    # Convert to list of [timestamp, value]
    return [(int(ts), float(val)) for ts, val in values]

def prepare_batch_data():
    """Prepare batch data from all three metrics"""
    print("Loading data from JSON files...")
    
    # Load all metrics
    cpu_data = load_metric_data('cluster_cpu_request_ratio.json')
    mem_data = load_metric_data('cluster_mem_request_ratio.json')
    pod_data = load_metric_data('cluster_pod_ratio.json')
    
    # Find common timestamps
    cpu_times = {ts for ts, _ in cpu_data}
    mem_times = {ts for ts, _ in mem_data}
    pod_times = {ts for ts, _ in pod_data}
    
    common_times = sorted(cpu_times & mem_times & pod_times)
    
    print(f"Found {len(common_times)} common timestamps")
    
    # Create dictionaries for quick lookup
    cpu_dict = dict(cpu_data)
    mem_dict = dict(mem_data)
    pod_dict = dict(pod_data)
    
    # Build batch data
    batch = []
    for ts in common_times:
        batch.append({
            "timestamp": datetime.fromtimestamp(ts).isoformat(),
            "cluster_cpu_request_ratio": cpu_dict[ts],
            "cluster_mem_request_ratio": mem_dict[ts],
            "cluster_pod_ratio": pod_dict[ts]
        })
    
    return batch

def test_batch_prediction(limit=None):
    """Test batch prediction endpoint"""
    print("\n" + "="*70)
    print("  BATCH PREDICTION TEST")
    print("="*70 + "\n")
    
    # Prepare data
    batch_data = prepare_batch_data()
    
    if limit:
        print(f"Using first {limit} samples for testing\n")
        batch_data = batch_data[:limit]
    else:
        print(f"Using all {len(batch_data)} samples\n")
    
    # Show sample data
    print("Sample data (first 3 records):")
    for i, record in enumerate(batch_data[:3], 1):
        print(f"\n  Record {i}:")
        print(f"    Timestamp: {record['timestamp']}")
        print(f"    CPU: {record['cluster_cpu_request_ratio']:.4f}")
        print(f"    Memory: {record['cluster_mem_request_ratio']:.4f}")
        print(f"    Pods: {record['cluster_pod_ratio']:.4f}")
    
    # Send request
    print(f"\nSending batch prediction request with {len(batch_data)} samples...")
    print("This may take a moment...\n")
    
    try:
        response = requests.post(
            f"{API_URL}/batch_predict",
            json={"samples": batch_data},
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes timeout
        )
        
        print(f"Status Code: {response.status_code}\n")
        
        if response.status_code == 200:
            result = response.json()
            
            # Summary statistics
            predictions = result.get('predictions', [])
            total = len(predictions)
            anomalies = sum(1 for p in predictions if p.get('is_anomaly', False))
            normal = total - anomalies
            
            print("=" * 70)
            print("  RESULTS SUMMARY")
            print("=" * 70)
            print(f"Total samples processed: {total}")
            print(f"Normal samples: {normal} ({normal/total*100:.1f}%)")
            print(f"Anomalies detected: {anomalies} ({anomalies/total*100:.1f}%)")
            print(f"Processing time: {result.get('processing_time', 'N/A')}")
            
            # Show anomalies
            if anomalies > 0:
                print("\n" + "=" * 70)
                print("  DETECTED ANOMALIES")
                print("=" * 70)
                
                anomaly_list = [p for p in predictions if p.get('is_anomaly', False)]
                for i, anom in enumerate(anomaly_list[:10], 1):  # Show first 10
                    print(f"\nAnomaly {i}:")
                    print(f"  Timestamp: {anom.get('timestamp')}")
                    print(f"  Anomaly Score: {anom.get('anomaly_score', 'N/A'):.4f}")
                    input_data = anom.get('input', {})
                    print(f"  CPU Ratio: {input_data.get('cluster_cpu_request_ratio', 'N/A'):.4f}")
                    print(f"  Memory Ratio: {input_data.get('cluster_mem_request_ratio', 'N/A'):.4f}")
                    print(f"  Pod Ratio: {input_data.get('cluster_pod_ratio', 'N/A'):.4f}")
                
                if anomalies > 10:
                    print(f"\n... and {anomalies - 10} more anomalies")
            
            # Alert info
            if result.get('alert_sent'):
                print("\n" + "=" * 70)
                print("  ALERT STATUS")
                print("=" * 70)
                print(f"Alert sent: YES")
                print(f"Alert summary: {result.get('alert_summary', {}).get('anomaly_count')} anomalies")
            
            return True
            
        else:
            print("ERROR:")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except:
                print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  AWS ANOMALY DETECTION - BATCH PREDICTION TEST")
    print("=" * 70)
    
    # Test with first 150 samples (enough for rolling windows)
    success = test_batch_prediction(limit=150)
    
    if success:
        print("\n" + "=" * 70)
        print("  TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("  TEST FAILED")
        print("=" * 70)
