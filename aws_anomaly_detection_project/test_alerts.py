"""
Test script for Alertmanager integration and email notifications
Tests the complete alert flow: API -> Alertmanager -> Email
"""

import requests
import json
import time
from datetime import datetime

API_URL = "http://localhost:5000"
ALERTMANAGER_URL = "http://localhost:9093"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def test_alertmanager_status():
    """Test Alertmanager is running"""
    print_section("1. ALERTMANAGER STATUS CHECK")
    
    try:
        response = requests.get(f"{ALERTMANAGER_URL}/api/v2/status")
        if response.status_code == 200:
            print(f"✅ Alertmanager is running")
            status = response.json()
            print(f"   Version: {status['versionInfo']['version']}")
            print(f"   Uptime: {status.get('uptime', 'N/A')}")
            return True
        else:
            print(f"❌ Alertmanager returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Alertmanager: {str(e)}")
        return False

def test_send_test_alert():
    """Send a test alert directly to Alertmanager"""
    print_section("2. SEND TEST ALERT TO ALERTMANAGER")
    
    test_alert = [{
        "labels": {
            "alertname": "TestAlert",
            "severity": "info",
            "service": "anomaly-detection-test",
            "instance": "test-instance"
        },
        "annotations": {
            "summary": "Test Alert from Python Script",
            "description": "This is a test alert to verify Alertmanager and email integration"
        },
        "startsAt": datetime.utcnow().isoformat() + "Z"
    }]
    
    try:
        response = requests.post(
            f"{ALERTMANAGER_URL}/api/v2/alerts",
            json=test_alert,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Test alert sent successfully to Alertmanager")
            print(f"   Alert: TestAlert (severity: info)")
            print(f"   Timestamp: {test_alert[0]['startsAt']}")
            return True
        else:
            print(f"❌ Failed to send alert. Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error sending alert: {str(e)}")
        return False

def check_active_alerts():
    """Check active alerts in Alertmanager"""
    print_section("3. CHECK ACTIVE ALERTS")
    
    try:
        response = requests.get(f"{ALERTMANAGER_URL}/api/v2/alerts")
        if response.status_code == 200:
            alerts = response.json()
            print(f"📊 Found {len(alerts)} active alert(s)")
            
            if len(alerts) > 0:
                for i, alert in enumerate(alerts[:5], 1):  # Show first 5
                    labels = alert.get('labels', {})
                    annotations = alert.get('annotations', {})
                    status = alert.get('status', {})
                    
                    print(f"\n   Alert {i}:")
                    print(f"   - Name: {labels.get('alertname', 'N/A')}")
                    print(f"   - Severity: {labels.get('severity', 'N/A')}")
                    print(f"   - State: {status.get('state', 'N/A')}")
                    print(f"   - Summary: {annotations.get('summary', 'N/A')}")
            
            return True
        else:
            print(f"❌ Failed to get alerts. Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking alerts: {str(e)}")
        return False

def test_anomaly_prediction_with_alert():
    """Test anomaly detection that should trigger an alert"""
    print_section("4. TEST ANOMALY DETECTION WITH ALERT")
    
    # Create data with anomalous values
    anomalous_data = {
        "samples": [
            {
                "timestamp": datetime.now().isoformat(),
                "cluster_cpu_request_ratio": 0.95,
                "cluster_mem_request_ratio": 0.98,
                "cluster_pod_ratio": 0.92
            },
            {
                "timestamp": datetime.now().isoformat(),
                "cluster_cpu_request_ratio": 0.93,
                "cluster_mem_request_ratio": 0.96,
                "cluster_pod_ratio": 0.90
            },
            {
                "timestamp": datetime.now().isoformat(),
                "cluster_cpu_request_ratio": 0.97,
                "cluster_mem_request_ratio": 0.99,
                "cluster_pod_ratio": 0.94
            }
        ]
    }
    
    print("Sending anomalous metrics to API...")
    print("Input Data (High resource usage):")
    print(f"  - CPU: {anomalous_data['samples'][0]['cluster_cpu_request_ratio']*100:.0f}%")
    print(f"  - Memory: {anomalous_data['samples'][0]['cluster_mem_request_ratio']*100:.0f}%")
    print(f"  - Pods: {anomalous_data['samples'][0]['cluster_pod_ratio']*100:.0f}%")
    
    try:
        response = requests.post(
            f"{API_URL}/batch_predict",
            json=anomalous_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get('summary', {})
            
            print(f"\n✅ Prediction successful!")
            print(f"   Total samples: {summary.get('total', 'N/A')}")
            print(f"   Anomalies detected: {summary.get('anomalies', 0)}")
            print(f"   Anomaly rate: {summary.get('anomaly_rate', 0)}%")
            
            if summary.get('anomalies', 0) > 0:
                print(f"\n⚠️  ANOMALY DETECTED!")
                print(f"   → Alert should be sent to Alertmanager")
                print(f"   → Email should be sent to configured recipients")
            
            return True
        else:
            print(f"❌ Prediction failed. Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return False

def check_alert_after_prediction():
    """Wait and check if alert was received by Alertmanager"""
    print_section("5. VERIFY ALERT RECEIVED BY ALERTMANAGER")
    
    print("⏳ Waiting 5 seconds for alert to be processed...")
    time.sleep(5)
    
    try:
        response = requests.get(f"{ALERTMANAGER_URL}/api/v2/alerts")
        if response.status_code == 200:
            alerts = response.json()
            
            # Look for anomaly alerts
            anomaly_alerts = [a for a in alerts if 'AzureClusterAnomaly' in a.get('labels', {}).get('alertname', '')]
            
            if anomaly_alerts:
                print(f"✅ Found {len(anomaly_alerts)} anomaly alert(s) in Alertmanager!")
                
                for alert in anomaly_alerts[:3]:
                    labels = alert.get('labels', {})
                    annotations = alert.get('annotations', {})
                    status = alert.get('status', {})
                    
                    print(f"\n   📧 Alert Details:")
                    print(f"   - Name: {labels.get('alertname', 'N/A')}")
                    print(f"   - Severity: {labels.get('severity', 'N/A')}")
                    print(f"   - State: {status.get('state', 'N/A')}")
                    print(f"   - CPU: {labels.get('cluster_cpu_request_ratio', 'N/A')}")
                    print(f"   - Memory: {labels.get('cluster_mem_request_ratio', 'N/A')}")
                    print(f"   - Summary: {annotations.get('summary', 'N/A')[:80]}")
                
                return True
            else:
                print(f"⚠️  No anomaly alerts found yet")
                print(f"   Total alerts: {len(alerts)}")
                return False
        else:
            print(f"❌ Failed to check alerts")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Run all alert tests"""
    print("\n" + "="*70)
    print("  🚨 ALERTMANAGER & EMAIL NOTIFICATION TEST SUITE 🚨")
    print("="*70)
    
    results = {
        'alertmanager_status': False,
        'test_alert': False,
        'active_alerts': False,
        'anomaly_prediction': False,
        'alert_verification': False
    }
    
    # Test 1: Alertmanager status
    results['alertmanager_status'] = test_alertmanager_status()
    
    if not results['alertmanager_status']:
        print("\n❌ Cannot continue - Alertmanager is not running")
        return
    
    # Test 2: Send test alert
    results['test_alert'] = test_send_test_alert()
    
    # Test 3: Check active alerts
    time.sleep(2)  # Wait a bit
    results['active_alerts'] = check_active_alerts()
    
    # Test 4: Test anomaly detection
    results['anomaly_prediction'] = test_anomaly_prediction_with_alert()
    
    # Test 5: Verify alert was received
    if results['anomaly_prediction']:
        results['alert_verification'] = check_alert_after_prediction()
    
    # Final summary
    print_section("FINAL SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"Tests passed: {passed_tests}/{total_tests}\n")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name.replace('_', ' ').title()}")
    
    print("\n" + "="*70)
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("\n📧 Email notifications should be sent to:")
        print("   - mohamedamine.chaabani@esprit.tn")
        print("   - aminchaabeni2000@gmail.com")
        print("\n💡 Check your inbox for alert emails!")
    else:
        print("⚠️  Some tests failed. Check the logs above.")
    
    print("="*70)

if __name__ == "__main__":
    main()
