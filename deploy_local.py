#!/usr/bin/env python3
"""
Local Deployment Script for Fraud Detection App
Stops existing container, rebuilds image, and deploys new version.
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, check=True, capture_output=False):
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
            return result.stdout.strip(), result.stderr.strip()
        else:
            subprocess.run(command, shell=True, check=check)
            return None, None
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error: {e}")
        if check:
            raise
        return None, str(e)


def get_container_info():
    """Get information about the currently running container."""
    logger.info("Getting container information...")
    
    # Get running containers
    stdout, stderr = run_command("docker ps", capture_output=True)
    
    if stderr:
        logger.error(f"Error getting container info: {stderr}")
        return None
    
    logger.info("Current running containers:")
    logger.info(stdout)
    
    # Check if our container is running
    if "fraud-dashboard" in stdout:
        logger.info("Found running fraud-dashboard container")
        return True
    else:
        logger.info("No fraud-dashboard container currently running")
        return False


def stop_existing_container():
    """Stop the existing container and remove it."""
    logger.info("Stopping existing container...")
    
    # Stop the container using docker-compose
    run_command("docker-compose down")
    
    # Also stop any containers with the old image SHA
    old_image_sha = "sha256:19b2ab07ca00cb2b3f1a98a66033e23a1e3de2a51bd1480c8237d075cb9b7c4d"
    
    # Find containers using the old image
    stdout, stderr = run_command(f"docker ps -a --filter ancestor={old_image_sha} --format '{{{{.ID}}}}'", capture_output=True)
    
    if stdout.strip():
        container_ids = stdout.strip().split('\n')
        for container_id in container_ids:
            if container_id:
                logger.info(f"Stopping container {container_id}")
                run_command(f"docker stop {container_id}", check=False)
                run_command(f"docker rm {container_id}", check=False)
    
    logger.info("Existing container stopped and removed")


def clean_docker_images():
    """Clean up old Docker images."""
    logger.info("Cleaning up old Docker images...")
    
    # Remove dangling images
    run_command("docker image prune -f", check=False)
    
    # Remove old fraud-dashboard images
    stdout, stderr = run_command("docker images --filter reference='*fraud-dashboard*' --format '{{.ID}}'", capture_output=True)
    
    if stdout.strip():
        image_ids = stdout.strip().split('\n')
        for image_id in image_ids:
            if image_id:
                logger.info(f"Removing old image {image_id}")
                run_command(f"docker rmi {image_id}", check=False)
    
    logger.info("Docker images cleaned up")


def build_new_image():
    """Build a new Docker image."""
    logger.info("Building new Docker image...")
    
    # Build the image using docker-compose
    run_command("docker-compose build --no-cache")
    
    logger.info("New Docker image built successfully")


def deploy_new_container():
    """Deploy the new container."""
    logger.info("Deploying new container...")
    
    # Start the new container
    run_command("docker-compose up -d")
    
    logger.info("New container deployed successfully")


def wait_for_health_check():
    """Wait for the container to be healthy."""
    logger.info("Waiting for container health check...")
    
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Check container status
            stdout, stderr = run_command("docker ps --filter name=fraud-dashboard --format '{{.Status}}'", capture_output=True)
            
            if "healthy" in stdout:
                logger.info("Container is healthy!")
                return True
            elif "unhealthy" in stdout:
                logger.error("Container is unhealthy!")
                return False
            else:
                logger.info(f"Container starting... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(10)
                attempt += 1
                
        except Exception as e:
            logger.error(f"Error checking container health: {e}")
            time.sleep(10)
            attempt += 1
    
    logger.error("Container failed to become healthy within timeout")
    return False


def test_endpoint():
    """Test the application endpoint."""
    logger.info("Testing application endpoint...")
    
    try:
        # Wait a bit for the app to fully start
        time.sleep(5)
        
        # Test the health endpoint
        stdout, stderr = run_command("curl -f http://localhost:8000/api/health", capture_output=True)
        
        if stdout:
            logger.info("Health endpoint response:")
            logger.info(stdout)
            return True
        else:
            logger.error("Health endpoint test failed")
            return False
            
    except Exception as e:
        logger.error(f"Error testing endpoint: {e}")
        return False


def show_container_logs():
    """Show container logs for debugging."""
    logger.info("Container logs:")
    run_command("docker-compose logs --tail=20", check=False)


def main():
    """Main deployment process."""
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION APP LOCAL DEPLOYMENT")
    logger.info("=" * 60)
    
    try:
        # Check if we're in the right directory
        if not os.path.exists("docker-compose.yml"):
            logger.error("docker-compose.yml not found. Please run this script from the project root.")
            sys.exit(1)
        
        if not os.path.exists("Dockerfile"):
            logger.error("Dockerfile not found. Please run this script from the project root.")
            sys.exit(1)
        
        # Step 1: Get container info
        container_running = get_container_info()
        
        # Step 2: Stop existing container
        if container_running:
            stop_existing_container()
        
        # Step 3: Clean up old images
        clean_docker_images()
        
        # Step 4: Build new image
        build_new_image()
        
        # Step 5: Deploy new container
        deploy_new_container()
        
        # Step 6: Wait for health check
        if not wait_for_health_check():
            logger.error("Deployment failed - container not healthy")
            show_container_logs()
            sys.exit(1)
        
        # Step 7: Test endpoint
        if not test_endpoint():
            logger.error("Deployment failed - endpoint test failed")
            show_container_logs()
            sys.exit(1)
        
        logger.info("=" * 60)
        logger.info("DEPLOYMENT SUCCESSFUL!")
        logger.info("Application is running at: http://localhost:8000")
        logger.info("Health endpoint: http://localhost:8000/api/health")
        logger.info("=" * 60)
        
        # Show final container status
        logger.info("Final container status:")
        run_command("docker ps --filter name=fraud-dashboard", check=False)
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        show_container_logs()
        sys.exit(1)


if __name__ == "__main__":
    main() 