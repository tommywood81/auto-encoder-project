#!/usr/bin/env python3
"""
Local Deployment for Fraud Detection Model.
Simple local testing and inference deployment.
"""

import os
import sys
import subprocess
import logging
import argparse
import json
import time
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalDeployment:
    """Local deployment for fraud detection model testing."""
    
    def __init__(self, port=5000):
        self.port = port
        self.project_name = "fraud-detection-api"
        self.container_name = f"{self.project_name}-local"
        self.docker_image_name = f"{self.project_name}:local"
        
    def run_command(self, command, check=True, capture_output=False):
        """Run a shell command and handle errors."""
        logger.info(f"Running: {command}")
        
        try:
            if capture_output:
                result = subprocess.run(
                    command, 
                    shell=True, 
                    check=check, 
                    capture_output=True, 
                    text=True
                )
                return result
            else:
                result = subprocess.run(command, shell=True, check=check)
                return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if check:
                raise
            return e
    
    def check_prerequisites(self):
        """Check if Docker is available."""
        logger.info("Checking prerequisites...")
        
        try:
            self.run_command("docker --version", capture_output=True)
            logger.info("Docker is installed")
            
            self.run_command("docker info", capture_output=True)
            logger.info("Docker daemon is running")
            
            return True
        except Exception as e:
            logger.error(f"Docker check failed: {e}")
            return False
    
    def build_docker_image(self):
        """Build the Docker image for local testing."""
        logger.info("Building Docker image for local testing...")
        
        try:
            # Build the image
            build_command = f"docker build -t {self.docker_image_name} ."
            self.run_command(build_command)
            
            logger.info(f"Docker image built: {self.docker_image_name}")
            return True
        except Exception as e:
            logger.error(f"Docker build failed: {e}")
            return False
    
    def start_local_server(self):
        """Start the local server for testing."""
        logger.info(f"Starting local server on port {self.port}...")
        
        try:
            # Stop any existing container
            self.run_command(f"docker stop {self.container_name}", check=False)
            self.run_command(f"docker rm {self.container_name}", check=False)
            
            # Run container
            run_command = f"docker run -d --name {self.container_name} -p {self.port}:5000 {self.docker_image_name}"
            self.run_command(run_command)
            
            # Wait for container to start
            logger.info("Waiting for server to start...")
            time.sleep(15)
            
            # Wait for health endpoint to be ready with retries
            max_retries = 10
            retry_delay = 3
            for attempt in range(max_retries):
                logger.info(f"Health check attempt {attempt + 1}/{max_retries}")
                if self.test_health_endpoint():
                    break
                elif attempt < max_retries - 1:
                    logger.info(f"Health check failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Health check failed after all retries")
                    return False
            
            # Test health endpoint one more time to confirm
            if self.test_health_endpoint():
                logger.info(f"Local server started successfully!")
                logger.info(f"Web interface: http://localhost:{self.port}")
                logger.info(f"API endpoint: http://localhost:{self.port}/health")
                return True
            else:
                logger.error("Health check failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start local server: {e}")
            return False
    
    def test_health_endpoint(self):
        """Test the health endpoint."""
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Health check passed: {health_data}")
                return True
            else:
                logger.error(f"Health check failed with status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during health check: {e}")
            return False
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout during health check: {e}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def test_model_inference(self):
        """Test model inference with sample data."""
        logger.info("Testing model inference...")
        
        try:
            # Get available dates
            response = requests.get(f"http://localhost:{self.port}/available-dates", timeout=30)
            if response.status_code != 200:
                logger.error("Failed to get available dates")
                return False
            
            dates_data = response.json()
            if not dates_data.get('dates'):
                logger.error("No available dates")
                return False
            
            # Test date analysis with first available date
            test_date = dates_data['dates'][0]
            logger.info(f"Testing analysis for date: {test_date}")
            
            analysis_response = requests.post(
                f"http://localhost:{self.port}/analyze-date",
                json={"date": test_date},
                timeout=30
            )
            
            if analysis_response.status_code == 200:
                analysis = analysis_response.json()
                logger.info(f"Date analysis test passed:")
                logger.info(f"  - Total transactions: {analysis['total_transactions']}")
                logger.info(f"  - Flagged transactions: {analysis['flagged_transactions']}")
                logger.info(f"  - AUC-ROC: {analysis['auc_roc']:.3f}")
                return True
            else:
                logger.error(f"Date analysis test failed: {analysis_response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Model inference test failed: {e}")
            return False
    
    def test_model_info(self):
        """Test model information endpoint."""
        logger.info("Testing model info endpoint...")
        
        try:
            response = requests.get(f"http://localhost:{self.port}/model-info", timeout=30)
            if response.status_code == 200:
                model_info = response.json()
                logger.info(f"Model info test passed:")
                logger.info(f"  - Model type: {model_info['model_type']}")
                logger.info(f"  - Strategy: {model_info['strategy']}")
                logger.info(f"  - Features: {model_info['feature_count']}")
                logger.info(f"  - Threshold: {model_info['threshold']}")
                return True
            else:
                logger.error(f"Model info test failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Model info test failed: {e}")
            return False
    
    def stop_local_server(self):
        """Stop the local server."""
        logger.info("Stopping local server...")
        
        try:
            self.run_command(f"docker stop {self.container_name}", check=False)
            self.run_command(f"docker rm {self.container_name}", check=False)
            logger.info("Local server stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop local server: {e}")
            return False
    
    def run_local_deployment(self):
        """Run the complete local deployment process."""
        logger.info("Starting Local Deployment for Fraud Detection Model")
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed")
            return False
        
        # Build Docker image
        if not self.build_docker_image():
            logger.error("Docker build failed")
            return False
        
        # Start local server
        if not self.start_local_server():
            logger.error("Failed to start local server")
            return False
        
        # Test endpoints
        logger.info("Testing API endpoints...")
        
        if not self.test_health_endpoint():
            logger.error("Health endpoint test failed")
            return False
        
        if not self.test_model_info():
            logger.error("Model info test failed")
            return False
        
        if not self.test_model_inference():
            logger.error("Model inference test failed")
            return False
        
        logger.info("✅ All tests passed! Local deployment successful.")
        logger.info(f"🌐 Web Dashboard: http://localhost:{self.port}")
        logger.info(f"🔧 API Health: http://localhost:{self.port}/health")
        logger.info(f"📊 Model Info: http://localhost:{self.port}/model-info")
        
        return True


def main():
    """Main function for local deployment."""
    parser = argparse.ArgumentParser(description="Local deployment for fraud detection model")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--stop", action="store_true", help="Stop the local server")
    
    args = parser.parse_args()
    
    deployment = LocalDeployment(port=args.port)
    
    if args.stop:
        deployment.stop_local_server()
    else:
        success = deployment.run_local_deployment()
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main() 