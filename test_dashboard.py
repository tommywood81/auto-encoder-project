#!/usr/bin/env python3
"""
Test script for the Fraud Detection Dashboard
"""

import requests
import json
import time

def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_prediction(threshold):
    """Test the prediction endpoint with a given threshold."""
    try:
        print(f"🧪 Testing threshold: {threshold}")
        print("   ⏳ Processing complete test set with real model...")
        
        # Use much longer timeout for complete test set processing
        response = requests.post(
            "http://localhost:8000/api/predict/all",
            json={"threshold": threshold},
            timeout=300  # 5 minutes timeout for complete test set
        )
        
        if response.status_code == 200:
            result = response.json()
            metrics = result['metrics']
            
            print(f"   ✅ Prediction successful")
            print(f"   📊 Total transactions: {metrics['total_transactions']:,}")
            print(f"   🚨 Fraud detected: {metrics['fraud_detected']:,}")
            print(f"   ✅ Normal transactions: {metrics['normal_transactions']:,}")
            print(f"   📈 Fraud rate: {metrics['fraud_rate']:.2%}")
            print(f"   ⏱️  Processing time: {metrics['processing_time_ms']:.2f}ms")
            
            # Show sample prediction
            if result['predictions']:
                sample = result['predictions'][0]
                print(f"   📋 Sample prediction: {sample['transaction_id']} - {sample['fraud_probability']:.3f}")
            
            return True
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Prediction timed out (dataset too large)")
        return False
    except Exception as e:
        print(f"   ❌ Prediction test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Fraud Detection Dashboard")
    print("=" * 40)
    
    # Test health
    if not test_health():
        print("\n❌ Health check failed. Dashboard may not be running.")
        return
    
    print()
    
    # Test predictions with different thresholds
    thresholds = [0.3, 0.5, 0.7]
    all_passed = True
    
    for threshold in thresholds:
        if not test_prediction(threshold):
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed.")
    
    print("\n🌐 Dashboard is available at: http://localhost:8000")
    print("📊 API documentation at: http://localhost:8000/api/docs")

if __name__ == "__main__":
    main() 