#!/usr/bin/env python3
"""
Local Deployment Pipeline for Fraud Detection Model.
Deploys the trained model to Digital Ocean droplet using local Docker build.
"""

import os
import sys
import subprocess
import logging
import argparse
import json
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalDeploymentPipeline:
    """Local deployment pipeline for fraud detection model."""
    
    def __init__(self, config):
        self.config = config
        self.project_name = "fraud-detection-api"
        self.docker_image_name = f"{self.project_name}"
        self.docker_tag = config.get('docker_tag', 'latest')
        
    def run_command(self, command, check=True, capture_output=False):
        """Run a shell command and handle errors."""
        logger.info(f"Running command: {command}")
        
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
        """Check if all required tools are installed."""
        logger.info("Checking prerequisites...")
        
        # Check Docker
        try:
            self.run_command("docker --version", capture_output=True)
            logger.info("✅ Docker is installed")
        except:
            logger.error("❌ Docker is not installed or not accessible")
            return False
        
        # Check if we can build Docker images
        try:
            self.run_command("docker info", capture_output=True)
            logger.info("✅ Docker daemon is running")
        except:
            logger.error("❌ Docker daemon is not running")
            return False
        
        # Check SSH access to droplet
        try:
            ssh_test = f"ssh -o ConnectTimeout=10 -o BatchMode=yes {self.config['ssh_user']}@{self.config['droplet_ip']} 'echo SSH connection successful'"
            self.run_command(ssh_test, capture_output=True)
            logger.info("✅ SSH access to droplet is working")
        except:
            logger.error("❌ SSH access to droplet failed. Make sure SSH key is configured.")
            return False
        
        return True
    
    def build_docker_image(self):
        """Build the Docker image."""
        logger.info("Building Docker image...")
        
        # Build the image
        build_command = f"docker build -t {self.docker_image_name}:{self.docker_tag} ."
        self.run_command(build_command)
        
        logger.info(f"✅ Docker image built: {self.docker_image_name}:{self.docker_tag}")
        return True
    
    def test_docker_image(self):
        """Test the Docker image locally."""
        logger.info("Testing Docker image locally...")
        
        # Run container in background
        container_name = f"{self.project_name}-test"
        run_command = f"docker run -d --name {container_name} -p 5000:5000 {self.docker_image_name}:{self.docker_tag}"
        
        try:
            self.run_command(run_command)
            
            # Wait for container to start
            import time
            time.sleep(10)
            
            # Test health endpoint
            import requests
            try:
                response = requests.get("http://localhost:5000/health", timeout=30)
                if response.status_code == 200:
                    logger.info("✅ Docker image test successful")
                    return True
                else:
                    logger.error(f"❌ Health check failed: {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Health check failed: {e}")
                return False
            finally:
                # Clean up test container
                self.run_command(f"docker stop {container_name}", check=False)
                self.run_command(f"docker rm {container_name}", check=False)
                
        except Exception as e:
            logger.error(f"❌ Docker image test failed: {e}")
            return False
    
    def save_and_transfer_image(self):
        """Save Docker image and transfer to droplet."""
        logger.info("Saving Docker image and transferring to droplet...")
        
        # Save Docker image to tar file
        image_tar = f"{self.docker_image_name}-{self.docker_tag}.tar"
        save_command = f"docker save {self.docker_image_name}:{self.docker_tag} -o {image_tar}"
        self.run_command(save_command)
        
        # Compress the tar file
        gzip_command = f"gzip {image_tar}"
        self.run_command(gzip_command)
        
        # Transfer to droplet
        scp_command = f"scp {image_tar}.gz {self.config['ssh_user']}@{self.config['droplet_ip']}:/tmp/"
        self.run_command(scp_command)
        
        # Clean up local files
        os.remove(f"{image_tar}.gz")
        
        logger.info("✅ Docker image transferred to droplet")
        return True
    
    def deploy_to_droplet(self):
        """Deploy the application to Digital Ocean droplet."""
        logger.info("Deploying to Digital Ocean droplet...")
        
        # Create deployment script
        deployment_script = self._create_deployment_script()
        
        # Copy deployment script to droplet
        scp_command = f"scp {deployment_script} {self.config['ssh_user']}@{self.config['droplet_ip']}:/tmp/"
        self.run_command(scp_command)
        
        # Execute deployment script on droplet
        ssh_command = f"ssh {self.config['ssh_user']}@{self.config['droplet_ip']} 'bash /tmp/deploy.sh'"
        self.run_command(ssh_command)
        
        # Clean up local deployment script
        os.remove(deployment_script)
        
        logger.info("✅ Deployment completed successfully")
        return True
    
    def _create_deployment_script(self):
        """Create deployment script for the droplet."""
        script_content = f"""#!/bin/bash
set -e

echo "Starting deployment on Digital Ocean droplet..."

# Update system
echo "Updating system packages..."
sudo apt-get update -y

# Install Docker if not installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install Docker Compose if not installed
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Load Docker image
echo "Loading Docker image..."
docker load -i /tmp/{self.docker_image_name}-{self.docker_tag}.tar.gz

# Create application directory
APP_DIR="/opt/{self.project_name}"
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Create docker-compose.yml
cat > $APP_DIR/docker-compose.yml << 'EOF'
version: '3.8'

services:
  fraud-detection-api:
    image: {self.docker_image_name}:{self.docker_tag}
    container_name: {self.project_name}
    restart: unless-stopped
    ports:
      - "80:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  default:
    name: {self.project_name}-network
EOF

# Stop and remove existing container
echo "Stopping existing container..."
docker stop {self.project_name} || true
docker rm {self.project_name} || true

# Start the application
echo "Starting fraud detection API..."
cd $APP_DIR
docker-compose up -d

# Wait for application to start
echo "Waiting for application to start..."
sleep 30

# Test the application
echo "Testing application..."
if curl -f http://localhost/health; then
    echo "✅ Application deployed successfully!"
    echo "🌐 Access the application at: http://{self.config['droplet_ip']}"
else
    echo "❌ Application health check failed"
    exit 1
fi

# Clean up transferred files
rm -f /tmp/{self.docker_image_name}-{self.docker_tag}.tar.gz
rm -f /tmp/deploy.sh
"""
        
        script_path = "/tmp/deploy.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def run_full_deployment(self):
        """Run the complete deployment pipeline."""
        logger.info("🚀 Starting fraud detection model deployment...")
        
        try:
            # Step 1: Check prerequisites
            if not self.check_prerequisites():
                logger.error("❌ Prerequisites check failed")
                return False
            
            # Step 2: Build Docker image
            if not self.build_docker_image():
                logger.error("❌ Docker image build failed")
                return False
            
            # Step 3: Test Docker image locally
            if not self.test_docker_image():
                logger.error("❌ Docker image test failed")
                return False
            
            # Step 4: Save and transfer image
            if not self.save_and_transfer_image():
                logger.error("❌ Image transfer failed")
                return False
            
            # Step 5: Deploy to droplet
            if not self.deploy_to_droplet():
                logger.error("❌ Deployment to droplet failed")
                return False
            
            logger.info("🎉 Deployment completed successfully!")
            logger.info(f"🌐 Application URL: http://{self.config['droplet_ip']}")
            logger.info("📊 Access the fraud detection interface at the URL above")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy Fraud Detection Model to Digital Ocean (Local Build)")
    parser.add_argument("--config", type=str, default="deployment_config.json", 
                       help="Path to deployment configuration file")
    parser.add_argument("--docker-tag", type=str, default="latest",
                       help="Docker image tag")
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Creating default configuration file...")
        
        default_config = {
            "droplet_ip": "209.38.89.159",
            "ssh_user": "root",
            "docker_tag": args.docker_tag
        }
        
        with open(args.config, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Default configuration created: {args.config}")
        logger.info("Please update the configuration file with your settings before running again.")
        return
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Update docker tag if provided
    if args.docker_tag != "latest":
        config["docker_tag"] = args.docker_tag
    
    # Create and run deployment pipeline
    pipeline = LocalDeploymentPipeline(config)
    success = pipeline.run_full_deployment()
    
    if success:
        logger.info("🎉 Deployment completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 