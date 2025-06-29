#!/usr/bin/env python3
"""
Quick Deploy Script for Fraud Detection Model.
One-command deployment to Digital Ocean droplet.
"""

import os
import sys
import subprocess
import logging
import json
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, check=True):
    """Run a shell command and handle errors."""
    logger.info(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if check:
            raise
        return e

def check_prerequisites():
    """Check if all prerequisites are met."""
    logger.info("🔍 Checking prerequisites...")
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        logger.error("❌ app.py not found. Please run this script from the project root directory.")
        return False
    
    # Check if model files exist
    if not os.path.exists("models/"):
        logger.error("❌ models/ directory not found. Please ensure you have trained models.")
        return False
    
    # Check Docker
    try:
        run_command("docker --version")
        logger.info("✅ Docker is installed")
    except:
        logger.error("❌ Docker is not installed or not accessible")
        return False
    
    # Check if Docker daemon is running
    try:
        run_command("docker info")
        logger.info("✅ Docker daemon is running")
    except:
        logger.error("❌ Docker daemon is not running")
        return False
    
    return True

def create_config():
    """Create deployment configuration if it doesn't exist."""
    config_file = "deployment_config.json"
    
    if not os.path.exists(config_file):
        logger.info("📝 Creating deployment configuration...")
        
        config = {
            "droplet_ip": "209.38.89.159",
            "ssh_user": "root",
            "docker_tag": "latest"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Configuration created: {config_file}")
        logger.info("Please update the configuration if needed before continuing.")
        return True
    else:
        logger.info("✅ Configuration file already exists")
        return True

def test_ssh_connection():
    """Test SSH connection to the droplet."""
    logger.info("🔐 Testing SSH connection...")
    
    try:
        with open("deployment_config.json", 'r') as f:
            config = json.load(f)
        
        ssh_test = f"ssh -o ConnectTimeout=10 -o BatchMode=yes {config['ssh_user']}@{config['droplet_ip']} 'echo SSH connection successful'"
        run_command(ssh_test)
        logger.info("✅ SSH connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ SSH connection failed: {e}")
        logger.error("Please ensure your SSH key is configured for the droplet.")
        return False

def build_and_test():
    """Build and test the Docker image locally."""
    logger.info("🏗️ Building Docker image...")
    
    try:
        # Build image
        run_command("docker build -t fraud-detection-api:latest .")
        logger.info("✅ Docker image built successfully")
        
        # Test image locally
        logger.info("🧪 Testing Docker image locally...")
        run_command("docker run -d --name fraud-test -p 5000:5000 fraud-detection-api:latest")
        
        # Wait for container to start
        time.sleep(15)
        
        # Test health endpoint
        import requests
        try:
            response = requests.get("http://localhost:5000/health", timeout=30)
            if response.status_code == 200:
                logger.info("✅ Local test successful")
                return True
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
        finally:
            # Clean up test container
            run_command("docker stop fraud-test", check=False)
            run_command("docker rm fraud-test", check=False)
            
    except Exception as e:
        logger.error(f"❌ Build/test failed: {e}")
        return False

def deploy():
    """Deploy to Digital Ocean droplet."""
    logger.info("🚀 Deploying to Digital Ocean droplet...")
    
    try:
        # Run the deployment script
        run_command("python deploy_local.py")
        logger.info("✅ Deployment completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        return False

def main():
    """Main deployment function."""
    logger.info("🚀 Starting Quick Deploy for Fraud Detection Model")
    logger.info("=" * 60)
    
    try:
        # Step 1: Check prerequisites
        if not check_prerequisites():
            logger.error("❌ Prerequisites check failed")
            return False
        
        # Step 2: Create configuration
        if not create_config():
            logger.error("❌ Configuration creation failed")
            return False
        
        # Step 3: Test SSH connection
        if not test_ssh_connection():
            logger.error("❌ SSH connection test failed")
            return False
        
        # Step 4: Build and test locally
        if not build_and_test():
            logger.error("❌ Local build/test failed")
            return False
        
        # Step 5: Deploy to droplet
        if not deploy():
            logger.error("❌ Deployment failed")
            return False
        
        # Success!
        logger.info("=" * 60)
        logger.info("🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("🌐 Your fraud detection application is now available at:")
        logger.info("   http://209.38.89.159")
        logger.info("")
        logger.info("📊 Features available:")
        logger.info("   • Web interface for testing")
        logger.info("   • REST API for integration")
        logger.info("   • Real-time fraud detection")
        logger.info("   • Health monitoring")
        logger.info("")
        logger.info("📖 For more information, see DEPLOYMENT_README.md")
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Deployment interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 