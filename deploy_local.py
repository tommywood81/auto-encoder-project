#!/usr/bin/env python3
"""
Simple Local Deployment for Autoencoder Fraud Detection Demo Dashboard.
Clean, explainable, and interactive dashboard for fraud detection demonstration.
"""

import os
import sys
import subprocess
import logging
import time
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleLocalDeployment:
    def __init__(self, port=5000):
        self.port = port
        self.container_name = "fraud-demo-local"
        self.docker_image_name = "fraud-demo:local"

    def run_command(self, command, check=True):
        logger.info(f"Running: {command}")
        try:
            subprocess.run(command, shell=True, check=check)
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if check:
                raise

    def check_docker(self):
        try:
            self.run_command("docker --version")
            self.run_command("docker info")
            logger.info("Docker is ready")
            return True
        except Exception as e:
            logger.error(f"Docker check failed: {e}")
            return False

    def build_image(self):
        try:
            self.run_command(f"docker build -t {self.docker_image_name} .")
            logger.info(f"Docker image built: {self.docker_image_name}")
            return True
        except Exception as e:
            logger.error(f"Docker build failed: {e}")
            return False

    def start_server(self):
        logger.info(f"Starting demo server on port {self.port}...")
        try:
            self.run_command(f"docker stop {self.container_name}", check=False)
            self.run_command(f"docker rm {self.container_name}", check=False)
            self.run_command(f"docker run -d --name {self.container_name} -p {self.port}:5000 {self.docker_image_name}")
            logger.info("Waiting for server to start...")
            time.sleep(10)
            return self.test_health()
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def test_health(self):
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

    def stop_server(self):
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
        logger.info("Starting Autoencoder Fraud Detection Demo Deployment")
        logger.info("=" * 60)
        if not self.check_docker():
            return False
        if not self.build_image():
            return False
        if not self.start_server():
            return False
        logger.info("=" * 60)
        logger.info("Demo Dashboard Successfully Deployed!")
        logger.info(f"Dashboard: http://localhost:{self.port}")
        logger.info(f"Health: http://localhost:{self.port}/health")
        logger.info("=" * 60)
        logger.info("Use Ctrl+C to stop the demo")
        return True

def main():
    port = 5000
    if len(sys.argv) > 1 and sys.argv[1] == "--stop":
        SimpleLocalDeployment(port=port).stop_server()
    else:
        success = SimpleLocalDeployment(port=port).deploy()
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main() 