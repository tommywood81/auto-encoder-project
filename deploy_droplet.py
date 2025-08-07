#!/usr/bin/env python3
"""
Production Deployment Script for Fraud Detection Dashboard
Deploys the application to a remote droplet via Docker Hub.

This script:
1. Builds and tags the Docker image
2. Pushes to Docker Hub
3. Connects to the droplet via SSH
4. Stops/removes existing containers
5. Pulls the new image
6. Runs the updated container
7. Verifies deployment health

Prerequisites:
- Docker Hub account and login (docker login)
- Droplet with Docker installed
- SSH access to droplet
- Open ports (80/443 for production)
"""

import os
import sys
import subprocess
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DropletDeployer:
    """Production deployment manager for fraud detection dashboard."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize deployer with configuration."""
        self.config = config
        self.docker_image_name = config['docker']['image_name']
        self.docker_username = config['docker']['username']
        self.docker_tag = config['docker']['tag']
        self.full_image_name = f"{self.docker_username}/{self.docker_image_name}:{self.docker_tag}"
        
        self.droplet_ip = config['droplet']['ip']
        self.ssh_user = config['droplet']['ssh_user']
        self.app_port = config['droplet']['app_port']
        self.container_name = config['droplet']['container_name']
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate deployment configuration."""
        required_fields = [
            'docker.image_name', 'docker.username', 'docker.tag',
            'droplet.ip', 'droplet.ssh_user', 'droplet.app_port'
        ]
        
        for field in required_fields:
            keys = field.split('.')
            value = self.config
            for key in keys:
                if key not in value:
                    raise ValueError(f"Missing required config field: {field}")
                value = value[key]
            
            if not value:
                raise ValueError(f"Empty value for required config field: {field}")
    
    def run_command(self, command: str, check: bool = True, capture_output: bool = False) -> Optional[str]:
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
                return result.stdout.strip()
            else:
                subprocess.run(command, shell=True, check=check)
                return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error: {e}")
            if check:
                raise
            return None
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        logger.info("Checking deployment prerequisites...")
        
        # Check if we're in the right directory
        if not os.path.exists("Dockerfile"):
            logger.error("Dockerfile not found. Please run this script from the project root.")
            return False
        
        if not os.path.exists("docker-compose.yml"):
            logger.error("docker-compose.yml not found. Please run this script from the project root.")
            return False
        
        # Check Docker login
        try:
            self.run_command("docker info", check=True, capture_output=True)
            logger.info("Docker is running and accessible")
        except Exception as e:
            logger.error(f"Docker is not accessible: {e}")
            return False
        
        # Check SSH connectivity
        ssh_test_cmd = f"ssh {self.ssh_user}@{self.droplet_ip} whoami"
        try:
            result = subprocess.run(ssh_test_cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and "root" in result.stdout:
                logger.info("SSH connection to droplet successful")
            else:
                logger.error(f"SSH test failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Cannot connect to droplet via SSH: {e}")
            logger.error("Please ensure:")
            logger.error("1. Droplet IP is correct")
            logger.error("2. SSH key is properly configured")
            logger.error("3. Droplet is running and accessible")
            return False
        
        # Check if Docker is available on droplet
        logger.info("Skipping Docker check on droplet (will verify during deployment)")
        
        logger.info("All prerequisites met successfully")
        return True
    
    def build_and_tag_image(self) -> bool:
        """Build and tag the Docker image."""
        logger.info("Building and tagging Docker image...")
        
        try:
            # Build the image
            build_cmd = f"docker build -t {self.full_image_name} ."
            self.run_command(build_cmd, check=True)
            
            logger.info(f"Successfully built image: {self.full_image_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build image: {e}")
            return False
    
    def push_to_docker_hub(self) -> bool:
        """Push the image to Docker Hub."""
        logger.info("Pushing image to Docker Hub...")
        
        try:
            # Push the image
            push_cmd = f"docker push {self.full_image_name}"
            self.run_command(push_cmd, check=True)
            
            logger.info(f"Successfully pushed image to Docker Hub: {self.full_image_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to push image to Docker Hub: {e}")
            logger.error("Please ensure you are logged in to Docker Hub: docker login")
            return False
    
    def deploy_to_droplet(self) -> bool:
        """Deploy the application to the droplet."""
        logger.info("Deploying to droplet...")
        
        try:
            # Stop and remove existing container
            stop_cmd = f"ssh {self.ssh_user}@{self.droplet_ip} docker stop {self.container_name} || true"
            self.run_command(stop_cmd, check=False)
            
            remove_cmd = f"ssh {self.ssh_user}@{self.droplet_ip} docker rm {self.container_name} || true"
            self.run_command(remove_cmd, check=False)
            
            # Remove old images to save space
            cleanup_cmd = f"ssh {self.ssh_user}@{self.droplet_ip} docker image prune -f"
            self.run_command(cleanup_cmd, check=False)
            
            # Pull the new image
            pull_cmd = f"ssh {self.ssh_user}@{self.droplet_ip} docker pull {self.full_image_name}"
            self.run_command(pull_cmd, check=True)
            
            # Run the new container
            run_cmd = f"ssh {self.ssh_user}@{self.droplet_ip} docker run -d --name {self.container_name} --restart unless-stopped -p {self.app_port}:8000 -v /var/log/fraud-dashboard:/app/logs -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 {self.full_image_name}"
            self.run_command(run_cmd, check=True)
            
            logger.info(f"Successfully deployed container: {self.container_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy to droplet: {e}")
            return False
    
    def verify_deployment(self) -> bool:
        """Verify the deployment is healthy."""
        logger.info("Verifying deployment health...")
        
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Check if container is running
                status_cmd = f"ssh {self.ssh_user}@{self.droplet_ip} docker ps --filter name={self.container_name} --format '{{{{.Status}}}}'"
                status = self.run_command(status_cmd, check=True, capture_output=True)
                
                if status and "Up" in status:
                    logger.info("Container is running")
                    
                    # Test the health endpoint
                    health_cmd = f"ssh {self.ssh_user}@{self.droplet_ip} curl -f http://localhost:{self.app_port}/api/health"
                    health_response = self.run_command(health_cmd, check=True, capture_output=True)
                    
                    if health_response and "healthy" in health_response.lower():
                        logger.info("Application health check passed")
                        logger.info(f"Dashboard is accessible at: http://{self.droplet_ip}:{self.app_port}")
                        return True
                    else:
                        logger.warning(f"Health check failed, retrying... (attempt {attempt + 1}/{max_attempts})")
                
                else:
                    logger.warning(f"Container not running, retrying... (attempt {attempt + 1}/{max_attempts})")
                
                time.sleep(10)
                attempt += 1
                
            except Exception as e:
                logger.warning(f"Health check error, retrying... (attempt {attempt + 1}/{max_attempts}): {e}")
                time.sleep(10)
                attempt += 1
        
        logger.error("Deployment verification failed after maximum attempts")
        return False
    
    def show_deployment_info(self):
        """Show deployment information and next steps."""
        logger.info("=" * 60)
        logger.info("DEPLOYMENT COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Dashboard URL: http://{self.droplet_ip}:{self.app_port}")
        logger.info(f"Container Name: {self.container_name}")
        logger.info(f"Docker Image: {self.full_image_name}")
        logger.info("=" * 60)
        logger.info("Next Steps:")
        logger.info("1. Open the dashboard URL in your browser")
        logger.info("2. Test the 'Analyse Transactions' functionality")
        logger.info("3. Monitor logs: ssh to droplet and run 'docker logs {self.container_name}'")
        logger.info("4. Set up SSL/HTTPS for production use")
        logger.info("=" * 60)
    
    def deploy(self) -> bool:
        """Execute the complete deployment process."""
        logger.info("=" * 60)
        logger.info("FRAUD DETECTION DASHBOARD - PRODUCTION DEPLOYMENT")
        logger.info("=" * 60)
        
        try:
            # Step 1: Check prerequisites
            if not self.check_prerequisites():
                logger.error("Prerequisites check failed")
                return False
            
            # Step 2: Build and tag image
            if not self.build_and_tag_image():
                logger.error("Image build failed")
                return False
            
            # Step 3: Push to Docker Hub
            if not self.push_to_docker_hub():
                logger.error("Docker Hub push failed")
                return False
            
            # Step 4: Deploy to droplet
            if not self.deploy_to_droplet():
                logger.error("Droplet deployment failed")
                return False
            
            # Step 5: Verify deployment
            if not self.verify_deployment():
                logger.error("Deployment verification failed")
                return False
            
            # Step 6: Show deployment info
            self.show_deployment_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed with unexpected error: {e}")
            return False


def load_config(config_path: str = "configs/deployment_config.yaml") -> Dict[str, Any]:
    """Load deployment configuration from YAML file."""
    import yaml
    
    if not os.path.exists(config_path):
        # Create default configuration
        default_config = {
            'docker': {
                'image_name': 'fraud-detection-dashboard',
                'username': 'your-dockerhub-username',
                'tag': 'latest'
            },
            'droplet': {
                'ip': 'your-droplet-ip',
                'ssh_user': 'root',
                'app_port': '80',
                'container_name': 'fraud-dashboard'
            }
        }
        
        # Create configs directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created default configuration file: {config_path}")
        logger.info("Please edit this file with your actual deployment settings")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {config_path}")
    return config


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Fraud Detection Dashboard to Droplet")
    parser.add_argument(
        "--config", 
        default="configs/deployment_config.yaml",
        help="Path to deployment configuration file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without deploying"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create deployer
        deployer = DropletDeployer(config)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - Validating configuration only")
            if deployer.check_prerequisites():
                logger.info("Configuration validation successful")
                return 0
            else:
                logger.error("Configuration validation failed")
                return 1
        
        # Execute deployment
        success = deployer.deploy()
        
        if success:
            logger.info("Deployment completed successfully")
            return 0
        else:
            logger.error("Deployment failed")
            return 1
            
    except Exception as e:
        logger.error(f"Deployment script failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
