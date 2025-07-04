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
import threading
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
        # Use the same image as deploy_local.py for consistency
        self.docker_image_name = f"{self.project_name}:local"
        self.container_name = f"{self.project_name}-production"
        
    def run_command(self, command, check=True, capture_output=False, timeout=None):
        """Run a shell command and handle errors."""
        logger.info(f"Running: {command}")
        
        try:
            if capture_output:
                result = subprocess.run(
                    command, 
                    shell=True, 
                    check=check, 
                    capture_output=True, 
                    text=True,
                    timeout=timeout
                )
                if result.stdout:
                    logger.info(f"Command output: {result.stdout.strip()}")
                if result.stderr:
                    logger.warning(f"Command stderr: {result.stderr.strip()}")
                return result
            else:
                result = subprocess.run(command, shell=True, check=check, timeout=timeout)
                return result
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {timeout} seconds: {command}")
            if check:
                raise
            return e
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"Command stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"Command stderr: {e.stderr}")
            if check:
                raise
            return e
    
    def check_prerequisites(self):
        """Check if all required tools are installed."""
        logger.info("Checking prerequisites...")
        
        # Check Docker
        try:
            self.run_command("docker --version", capture_output=True, timeout=10)
            logger.info("Docker is installed")
            
            self.run_command("docker info", capture_output=True, timeout=10)
            logger.info("Docker daemon is running")
        except subprocess.TimeoutExpired:
            logger.error("Docker info command timed out. Docker Desktop may not be running.")
            logger.info("Please start Docker Desktop and try again.")
            return False
        except Exception as e:
            logger.error(f"Docker check failed: {e}")
            return False
        
        # Check SSH access to droplet
        try:
            ssh_test = f"ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no {self.config['ssh_user']}@{self.config['droplet_ip']} 'echo SSH connection successful'"
            self.run_command(ssh_test, capture_output=True)
            logger.info("SSH access to droplet is working")
        except Exception as e:
            logger.warning(f"SSH access to droplet failed: {e}")
            logger.warning("This is expected if SSH keys are not configured yet")
            logger.warning("The deployment will continue but may fail at the transfer step")
            # Don't return False here - let it continue for testing
        
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
            image_tar = "fraud-detection-api-local.tar"
            logger.info(f"Saving Docker image to {image_tar}...")
            save_command = f"docker save {self.docker_image_name} -o {image_tar}"
            self.run_command(save_command)
            
            # Get file size for progress tracking
            file_size = os.path.getsize(image_tar)
            logger.info(f"Image saved: {file_size / (1024**3):.2f} GB")
            
            # Transfer to droplet with progress
            logger.info("Transferring image to droplet (this may take 15-30 minutes)...")
            logger.info("🔄 Transfer in progress... (You'll see progress updates from scp)")
            
            # Start a background thread to show transfer is still active
            transfer_active = True
            def progress_indicator():
                dots = 0
                while transfer_active:
                    time.sleep(10)  # Update every 10 seconds
                    dots = (dots + 1) % 4
                    logger.info(f"🔄 Transfer still active{'.' * dots}")
            
            progress_thread = threading.Thread(target=progress_indicator, daemon=True)
            progress_thread.start()
            
            try:
                scp_command = f"scp -v {image_tar} {self.config['ssh_user']}@{self.config['droplet_ip']}:/tmp/"
                self.run_command(scp_command)
            finally:
                transfer_active = False
            
            # Clean up local files
            os.remove(image_tar)
            
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
            logger.info("Creating deployment script...")
            deployment_script = self._create_deployment_script()
            logger.info(f"Deployment script created: {deployment_script}")
            
            # Copy deployment script to droplet
            logger.info("Copying deployment script to droplet...")
            scp_command = f"scp {deployment_script} {self.config['ssh_user']}@{self.config['droplet_ip']}:/tmp/"
            self.run_command(scp_command)
            
            # Verify script was copied
            logger.info("Verifying deployment script on droplet...")
            verify_command = f"ssh {self.config['ssh_user']}@{self.config['droplet_ip']} ls -la /tmp/deploy_script.sh"
            self.run_command(verify_command, capture_output=True)
            
            # Execute deployment script on droplet
            logger.info("Executing deployment script on droplet...")
            ssh_command = f"ssh {self.config['ssh_user']}@{self.config['droplet_ip']} bash /tmp/deploy_script.sh"
            self.run_command(ssh_command)
            
            # Clean up local deployment script
            logger.info("Cleaning up local deployment script...")
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

# Stop and remove existing container
echo "Stopping existing container..."
docker stop fraud-detection-api-production || true
docker rm fraud-detection-api-production || true

# Load Docker image
echo "Loading Docker image..."
docker load -i /tmp/fraud-detection-api-local.tar

# Run the container
echo "Starting fraud detection API..."
docker run -d \
  --name fraud-detection-api-production \
  --restart unless-stopped \
  -p 80:5000 \
  -e FLASK_ENV=production \
  fraud-detection-api:local

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
rm -f /tmp/fraud-detection-api-local.tar
rm -f /tmp/deploy_script.sh
"""
        script_path = "deploy_script.sh"
        with open(script_path, 'w', newline='\n') as f:
            f.write(script_content)
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
        logger.info("=" * 60)
        logger.info("STARTING PRODUCTION DEPLOYMENT FOR FRAUD DETECTION MODEL")
        logger.info("=" * 60)
        
        try:
            # Step 1: Check prerequisites
            logger.info("\n" + "=" * 40)
            logger.info("STEP 1: CHECKING PREREQUISITES")
            logger.info("=" * 40)
            if not self.check_prerequisites():
                logger.error("❌ Prerequisites check failed")
                return False
            logger.info("✅ Prerequisites check passed")
            
            # Step 2: Build production Docker image
            logger.info("\n" + "=" * 40)
            logger.info("STEP 2: BUILDING PRODUCTION DOCKER IMAGE")
            logger.info("=" * 40)
            if not self.build_production_image():
                logger.error("❌ Production image build failed")
                return False
            logger.info("✅ Production image built successfully")
            
            # Step 3: Test production image locally
            logger.info("\n" + "=" * 40)
            logger.info("STEP 3: TESTING PRODUCTION IMAGE LOCALLY")
            logger.info("=" * 40)
            if not self.test_production_image():
                logger.error("❌ Production image test failed")
                return False
            logger.info("✅ Production image test passed")
            
            # Step 4: Save and transfer image
            logger.info("\n" + "=" * 40)
            logger.info("STEP 4: SAVING AND TRANSFERRING IMAGE")
            logger.info("=" * 40)
            if not self.save_and_transfer_image():
                logger.error("❌ Image transfer failed")
                return False
            logger.info("✅ Image transferred successfully")
            
            # Step 5: Deploy to droplet
            logger.info("\n" + "=" * 40)
            logger.info("STEP 5: DEPLOYING TO DROPLET")
            logger.info("=" * 40)
            if not self.deploy_to_droplet():
                logger.error("❌ Deployment to droplet failed")
                return False
            logger.info("✅ Deployment to droplet completed")
            
            # Step 6: Verify deployment
            logger.info("\n" + "=" * 40)
            logger.info("STEP 6: VERIFYING PRODUCTION DEPLOYMENT")
            logger.info("=" * 40)
            if not self.verify_production_deployment():
                logger.error("❌ Production deployment verification failed")
                return False
            logger.info("✅ Production deployment verified")
            
            logger.info("\n" + "=" * 60)
            logger.info("🎉 PRODUCTION DEPLOYMENT COMPLETED SUCCESSFULLY! 🎉")
            logger.info("=" * 60)
            logger.info(f"🌐 Production URL: http://{self.config['droplet_ip']}")
            logger.info(f"🏥 API Health: http://{self.config['droplet_ip']}/health")
            logger.info(f"📚 API Docs: http://{self.config['droplet_ip']}/docs")
            logger.info("=" * 60)
            
            return True
            
        except KeyboardInterrupt:
            logger.info("\n⚠️  Deployment interrupted by user")
            return False
        except Exception as e:
            logger.error(f"💥 Production deployment failed: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Production Deployment for Fraud Detection Model")
    parser.add_argument("--config", type=str, default="deployment_config.json", 
                       help="Path to deployment configuration file")
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Creating default configuration file...")
        
        default_config = {
            "droplet_ip": "209.38.89.159",
            "ssh_user": "root"
        }
        
        with open(args.config, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Default configuration created: {args.config}")
        logger.info("Please update the configuration file with your settings before running again.")
        return
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
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