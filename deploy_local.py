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
            time.sleep(10)
            
            # Test health endpoint
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
            response = requests.get(f"http://localhost:{self.port}/health", timeout=30)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Health check passed: {health_data}")
                return True
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def test_model_inference(self):
        """Test model inference with sample data."""
        logger.info("Testing model inference...")
        
        try:
            # Get test data
            response = requests.get(f"http://localhost:{self.port}/test-data", timeout=30)
            if response.status_code != 200:
                logger.error("Failed to get test data")
                return False
            
            test_data = response.json()
            if not test_data.get('test_data'):
                logger.error("No test data available")
                return False
            
            # Test single prediction
            sample_features = test_data['test_data'][0]['features']
            prediction_response = requests.post(
                f"http://localhost:{self.port}/predict",
                json={"features": sample_features},
                timeout=30
            )
            
            if prediction_response.status_code == 200:
                prediction = prediction_response.json()
                logger.info(f"Single prediction test passed: {prediction}")
                
                # Test batch prediction
                batch_data = {"transactions": [item['features'] for item in test_data['test_data'][:3]]}
                batch_response = requests.post(
                    f"http://localhost:{self.port}/predict-batch",
                    json=batch_data,
                    timeout=30
                )
                
                if batch_response.status_code == 200:
                    batch_prediction = batch_response.json()
                    logger.info(f"Batch prediction test passed: {len(batch_prediction['predictions'])} predictions")
                    return True
                else:
                    logger.error("Batch prediction test failed")
                    return False
            else:
                logger.error("Single prediction test failed")
                return False
                
        except Exception as e:
            logger.error(f"Model inference test failed: {e}")
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
        
        try:
            # Step 1: Check prerequisites
            if not self.check_prerequisites():
                logger.error("Prerequisites check failed")
                return False
            
            # Step 2: Build Docker image
            if not self.build_docker_image():
                logger.error("Docker build failed")
                return False
            
            # Step 3: Start local server
            if not self.start_local_server():
                logger.error("Failed to start local server")
                return False
            
            # Step 4: Test model inference
            if not self.test_model_inference():
                logger.error("Model inference test failed")
                return False
            
            logger.info("LOCAL DEPLOYMENT COMPLETED SUCCESSFULLY!")
            logger.info(f"Access the web interface at: http://localhost:{self.port}")
            logger.info(f"API documentation at: http://localhost:{self.port}/docs")
            logger.info("Use Ctrl+C to stop the server when done testing")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("\nDeployment interrupted by user")
            self.stop_local_server()
            return False
        except Exception as e:
            logger.error(f"Local deployment failed: {e}")
            self.stop_local_server()
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Local Deployment for Fraud Detection Model")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--stop", action="store_true", help="Stop the local server")
    
    args = parser.parse_args()
    
    deployment = LocalDeployment(port=args.port)
    
    if args.stop:
        success = deployment.stop_local_server()
        sys.exit(0 if success else 1)
    else:
        success = deployment.run_local_deployment()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 