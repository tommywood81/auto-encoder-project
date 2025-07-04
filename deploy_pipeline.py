#!/usr/bin/env python3
"""
Production Deployment Pipeline for Fraud Detection Model.
Deploys the trained model to Digital Ocean droplet using Docker.
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

class ProductionDeployment:
    """Production deployment pipeline for fraud detection model."""
    
    def __init__(self, config):
        self.config = config
        self.project_name = "fraud-detection-api"
        self.docker_image_name = f"{self.project_name}:production"
        self.container_name = self.project_name
        
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
        """Check if all required tools are installed."""
        logger.info("Checking prerequisites...")
        
        # Check Docker
        try:
            self.run_command("docker --version", capture_output=True)
            logger.info("Docker is installed")
            
            self.run_command("docker info", capture_output=True)
            logger.info("Docker daemon is running")
        except Exception as e:
            logger.error(f"Docker check failed: {e}")
            return False
        
        # Check SSH access to droplet
        try:
            ssh_test = f"ssh -o ConnectTimeout=10 -o BatchMode=yes {self.config['ssh_user']}@{self.config['droplet_ip']} 'echo SSH connection successful'"
            self.run_command(ssh_test, capture_output=True)
            logger.info("SSH access to droplet is working")
        except Exception as e:
            logger.error(f"SSH access to droplet failed: {e}")
            logger.error("Make sure SSH key is configured and droplet is accessible")
            return False
        
        return True
    
    def build_production_image(self):
        """Build the production Docker image."""
        logger.info("Building production Docker image...")
        
        try:
            # Build the image
            build_command = f"docker build -t {self.docker_image_name} ."
            self.run_command(build_command)
            
            logger.info(f"Production Docker image built: {self.docker_image_name}")
            return True
        except Exception as e:
            logger.error(f"Docker build failed: {e}")
            return False
    
    def test_production_image(self):
        """Test the production Docker image locally."""
        logger.info("Testing production Docker image locally...")
        
        try:
            # Run container in background
            test_container = f"{self.container_name}-test"
            run_command = f"docker run -d --name {test_container} -p 5000:5000 {self.docker_image_name}"
            
            self.run_command(run_command)
            
            # Wait for container to start
            logger.info("Waiting for container to start...")
            time.sleep(15)
            
            # Test health endpoint
            try:
                response = requests.get("http://localhost:5000/health", timeout=30)
                if response.status_code == 200:
                    logger.info("Production image test successful")
                    return True
                else:
                    logger.error(f"Health check failed: {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                logger.error(f"Health check failed: {e}")
                return False
            finally:
                # Clean up test container
                self.run_command(f"docker stop {test_container}", check=False)
                self.run_command(f"docker rm {test_container}", check=False)
                
        except Exception as e:
            logger.error(f"Production image test failed: {e}")
            return False
    
    def save_and_transfer_image(self):
        """Save Docker image and transfer to droplet."""
        logger.info("Saving Docker image and transferring to droplet...")
        
        try:
            # Save Docker image to tar file
            image_tar = f"{self.docker_image_name.replace(':', '-')}.tar"
            save_command = f"docker save {self.docker_image_name} -o {image_tar}"
            self.run_command(save_command)
            
            # Compress the tar file
            gzip_command = f"gzip {image_tar}"
            self.run_command(gzip_command)
            
            # Transfer to droplet
            scp_command = f"scp {image_tar}.gz {self.config['ssh_user']}@{self.config['droplet_ip']}:/tmp/"
            self.run_command(scp_command)
            
            # Clean up local files
            os.remove(f"{image_tar}.gz")
            
            logger.info("Docker image transferred to droplet")
            return True
        except Exception as e:
            logger.error(f"Image transfer failed: {e}")
            return False
    
    def deploy_to_droplet(self):
        """Deploy the application to Digital Ocean droplet."""
        logger.info("Deploying to Digital Ocean droplet...")
        
        try:
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
            
            logger.info("Deployment completed successfully")
            return True
        except Exception as e:
            logger.error(f"Deployment to droplet failed: {e}")
            return False
    
    def _create_deployment_script(self):
        """Create deployment script for the droplet."""
        script_content = f"""#!/bin/bash
set -e

echo "Starting production deployment on Digital Ocean droplet..."

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
docker load -i /tmp/{self.docker_image_name.replace(':', '-')}.tar.gz

# Create application directory
APP_DIR="/opt/{self.project_name}"
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Create docker-compose.yml
cat > $APP_DIR/docker-compose.yml << 'EOF'
version: '3.8'

services:
  fraud-detection-api:
    image: {self.docker_image_name}
    container_name: {self.container_name}
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
docker stop {self.container_name} || true
docker rm {self.container_name} || true

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
    echo "Application deployed successfully!"
echo "Access the application at: http://{self.config['droplet_ip']}"
else
    echo "Application health check failed"
    exit 1
fi

# Clean up transferred files
rm -f /tmp/{self.docker_image_name.replace(':', '-')}.tar.gz
rm -f /tmp/deploy.sh
"""
        
        script_path = "/tmp/deploy.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def verify_production_deployment(self):
        """Verify the production deployment."""
        logger.info("Verifying production deployment...")
        
        try:
            # Test the production endpoint
            response = requests.get(f"http://{self.config['droplet_ip']}/health", timeout=30)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Production deployment verified: {health_data}")
                return True
            else:
                logger.error(f"Production health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Production verification failed: {e}")
            return False
    
    def run_production_deployment(self):
        """Run the complete production deployment process."""
        logger.info("Starting Production Deployment for Fraud Detection Model")
        
        try:
            # Step 1: Check prerequisites
            if not self.check_prerequisites():
                logger.error("Prerequisites check failed")
                return False
            
            # Step 2: Build production Docker image
            if not self.build_production_image():
                logger.error("Production image build failed")
                return False
            
            # Step 3: Test production image locally
            if not self.test_production_image():
                logger.error("Production image test failed")
                return False
            
            # Step 4: Save and transfer image
            if not self.save_and_transfer_image():
                logger.error("Image transfer failed")
                return False
            
            # Step 5: Deploy to droplet
            if not self.deploy_to_droplet():
                logger.error("Deployment to droplet failed")
                return False
            
            # Step 6: Verify deployment
            if not self.verify_production_deployment():
                logger.error("Production deployment verification failed")
                return False
            
            logger.info("PRODUCTION DEPLOYMENT COMPLETED SUCCESSFULLY!")
            logger.info(f"Production URL: http://{self.config['droplet_ip']}")
            logger.info(f"API Health: http://{self.config['droplet_ip']}/health")
            logger.info(f"API Docs: http://{self.config['droplet_ip']}/docs")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("\nDeployment interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Production Deployment for Fraud Detection Model")
    parser.add_argument("--config", type=str, default="deployment_config.json", 
                       help="Path to deployment configuration file")
    parser.add_argument("--docker-tag", type=str, default="production",
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
    if args.docker_tag != "production":
        config["docker_tag"] = args.docker_tag
    
    # Create and run deployment pipeline
    deployment = ProductionDeployment(config)
    success = deployment.run_production_deployment()
    
    if success:
        logger.info("Production deployment completed successfully!")
        sys.exit(0)
    else:
        logger.error("Production deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 