#!/usr/bin/env python3
"""
Simple Local Deployment for Autoencoder Fraud Detection Demo Dashboard.
Clean, explainable, and interactive dashboard for fraud detection demonstration.
"""

import os
import sys
import subprocess
import logging
import argparse
import time
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleLocalDeployment:
    """Simple local deployment for autoencoder fraud detection demo."""
    
    def __init__(self, port=5000):
        self.port = port
        self.container_name = "fraud-demo-local"
        self.docker_image_name = "fraud-demo:local"
        
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
    
    def check_docker(self):
        """Check if Docker is available."""
        logger.info("Checking Docker...")
        
        try:
            self.run_command("docker --version", capture_output=True)
            self.run_command("docker info", capture_output=True)
            logger.info("Docker is ready")
            return True
        except Exception as e:
            logger.error(f"Docker check failed: {e}")
            return False
    
    def build_image(self):
        """Build the Docker image."""
        logger.info("Building Docker image...")
        
        try:
            build_command = f"docker build -t {self.docker_image_name} ."
            self.run_command(build_command)
            logger.info(f"Docker image built: {self.docker_image_name}")
            return True
        except Exception as e:
            logger.error(f"Docker build failed: {e}")
            return False
    
    def start_server(self):
        """Start the local server."""
        logger.info(f"Starting demo server on port {self.port}...")
        
        try:
            # Stop any existing container
            self.run_command(f"docker stop {self.container_name}", check=False)
            self.run_command(f"docker rm {self.container_name}", check=False)
            
            # Run container
            run_command = f"docker run -d --name {self.container_name} -p {self.port}:5000 {self.docker_image_name}"
            self.run_command(run_command)
            
            # Wait for server to start
            logger.info("Waiting for server to start...")
            time.sleep(10)
            
            # Test health endpoint
            if self.test_health():
                logger.info("Demo server started successfully!")
                return True
            else:
                logger.error("Health check failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def test_health(self):
        """Test the health endpoint."""
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=10)
            if response.status_code == 200:
                logger.info("Health check passed")
                return True
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    def test_demo_features(self):
        """Test the demo dashboard features."""
        logger.info("Testing demo features...")
        
        tests = [
            ("Available dates", f"http://localhost:{self.port}/available-dates"),
            ("Model info", f"http://localhost:{self.port}/model-info"),
            ("Sample data", f"http://localhost:{self.port}/test-data"),
        ]
        
        all_passed = True
        for test_name, url in tests:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"{test_name} - OK")
                else:
                    logger.error(f"{test_name} - Failed ({response.status_code})")
                    all_passed = False
            except Exception as e:
                logger.error(f"{test_name} - Error: {e}")
                all_passed = False
        
        return all_passed
    
    def stop_server(self):
        """Stop the local server."""
        logger.info("Stopping demo server...")
        
        try:
            self.run_command(f"docker stop {self.container_name}", check=False)
            self.run_command(f"docker rm {self.container_name}", check=False)
            logger.info("Demo server stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
            return False
    
    def deploy(self):
        """Run the complete deployment process."""
        logger.info("Starting Autoencoder Fraud Detection Demo Deployment")
        logger.info("=" * 60)
        
        # Check Docker
        if not self.check_docker():
            return False
        
        # Build image
        if not self.build_image():
            return False
        
        # Start server
        if not self.start_server():
            return False
        
        # Test features
        if not self.test_demo_features():
            logger.warning("Some features failed, but server is running")
        
        # Success message
        logger.info("=" * 60)
        logger.info("Demo Dashboard Successfully Deployed!")
        logger.info("=" * 60)
        logger.info(f"Dashboard: http://localhost:{self.port}")
        logger.info(f"Health: http://localhost:{self.port}/health")
        logger.info("=" * 60)
        logger.info("Demo Features:")
        logger.info("   • Date Selector - Filter by specific dates")
        logger.info("   • Threshold Slider - Adjust fraud sensitivity (80-100)")
        logger.info("   • Metrics Panel - Real-time fraud detection stats")
        logger.info("   • Full Data Table - Every transaction with explainability")
        logger.info("   • Toggle Filters - View specific fraud categories")
        logger.info("   • Information Modal - Learn about the autoencoder model")
        logger.info("=" * 60)
        logger.info("Use Ctrl+C to stop the demo")
        
        return True


def main():
    """Main function for local deployment."""
    parser = argparse.ArgumentParser(
        description="Simple local deployment for autoencoder fraud detection demo"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000, 
        help="Port to run the demo server on (default: 5000)"
    )
    parser.add_argument(
        "--stop", 
        action="store_true", 
        help="Stop the demo server"
    )
    
    args = parser.parse_args()
    
    deployment = SimpleLocalDeployment(port=args.port)
    
    if args.stop:
        deployment.stop_server()
    else:
        success = deployment.deploy()
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main() 