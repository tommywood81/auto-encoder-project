#!/usr/bin/env python3
"""
Local Testing Script for Fraud Detection API.
Tests the API locally before deployment.
"""

import requests
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_endpoints(base_url="http://localhost:5000"):
    """Test all API endpoints."""
    
    logger.info(f"Testing API endpoints at {base_url}")
    
    # Test 1: Health Check
    logger.info("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Health check passed: {data}")
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Model Info
    logger.info("Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Model info: {data}")
        else:
            logger.error(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Model info failed: {e}")
        return False
    
    # Test 3: Get Test Data
    logger.info("Testing test data endpoint...")
    try:
        response = requests.get(f"{base_url}/test-data", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Test data retrieved: {data['count']} data points")
            
            # Use first data point for prediction test
            if data['test_data']:
                test_point = data['test_data'][0]
                logger.info(f"Using test point ID: {test_point['id']}")
                
                # Test 4: Single Prediction
                logger.info("Testing single prediction endpoint...")
                prediction_response = requests.post(
                    f"{base_url}/predict",
                    json={"features": test_point['features']},
                    timeout=10
                )
                
                if prediction_response.status_code == 200:
                    prediction_data = prediction_response.json()
                    logger.info(f"✅ Prediction successful: {prediction_data['prediction']}")
                else:
                    logger.error(f"❌ Prediction failed: {prediction_response.status_code}")
                    logger.error(f"Response: {prediction_response.text}")
                    return False
            else:
                logger.error("❌ No test data available")
                return False
        else:
            logger.error(f"❌ Test data failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Test data failed: {e}")
        return False
    
    # Test 5: Batch Prediction
    logger.info("Testing batch prediction endpoint...")
    try:
        response = requests.get(f"{base_url}/test-data", timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Use first 2 data points for batch test
            batch_data = {
                "transactions": [
                    {"features": data['test_data'][0]['features']},
                    {"features": data['test_data'][1]['features']}
                ]
            }
            
            batch_response = requests.post(
                f"{base_url}/predict-batch",
                json=batch_data,
                timeout=10
            )
            
            if batch_response.status_code == 200:
                batch_result = batch_response.json()
                logger.info(f"✅ Batch prediction successful: {batch_result['total_transactions']} transactions")
            else:
                logger.error(f"❌ Batch prediction failed: {batch_response.status_code}")
                return False
        else:
            logger.error(f"❌ Could not get test data for batch prediction")
            return False
    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {e}")
        return False
    
    logger.info("🎉 All API tests passed!")
    return True

def main():
    """Main test function."""
    logger.info("🧪 Starting local API tests...")
    
    # Wait a bit for the API to start
    logger.info("Waiting for API to start...")
    time.sleep(5)
    
    # Test the API
    success = test_api_endpoints()
    
    if success:
        logger.info("✅ All tests passed! API is ready for deployment.")
        return True
    else:
        logger.error("❌ Tests failed! Check the API logs.")
        return False

if __name__ == "__main__":
    main() 