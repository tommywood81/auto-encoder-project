#!/usr/bin/env python3
"""
Docker Hub Deployment Script for Fraud Detection Model.
Builds, tags, and pushes the model to Docker Hub for easy distribution.
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

def create_docker_hub_config():
    """Create Docker Hub configuration if it doesn't exist."""
    config_file = "docker_hub_config.json"
    
    if not os.path.exists(config_file):
        logger.info("📝 Creating Docker Hub configuration...")
        
        config = {
            "docker_hub_username": "your-username",
            "repository_name": "fraud-detection-api",
            "image_tag": "latest",
            "description": "AI-powered fraud detection using autoencoders"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Configuration created: {config_file}")
        logger.info("Please update the configuration with your Docker Hub username before continuing.")
        return False  # Return False to prompt user to update config
    else:
        logger.info("✅ Configuration file already exists")
        return True

def login_to_docker_hub():
    """Login to Docker Hub."""
    logger.info("🔐 Logging into Docker Hub...")
    
    try:
        # Check if already logged in
        result = run_command("docker info", check=False)
        if "Username:" in result.stdout:
            logger.info("✅ Already logged into Docker Hub")
            return True
        
        # Prompt for login
        logger.info("Please enter your Docker Hub credentials:")
        run_command("docker login")
        logger.info("✅ Successfully logged into Docker Hub")
        return True
    except Exception as e:
        logger.error(f"❌ Docker Hub login failed: {e}")
        return False

def build_image():
    """Build the Docker image."""
    logger.info("🏗️ Building Docker image...")
    
    try:
        with open("docker_hub_config.json", 'r') as f:
            config = json.load(f)
        
        username = config["docker_hub_username"]
        repo_name = config["repository_name"]
        tag = config["image_tag"]
        
        # Build image
        image_name = f"{username}/{repo_name}:{tag}"
        run_command(f"docker build -t {image_name} .")
        logger.info(f"✅ Docker image built successfully: {image_name}")
        
        # Also tag as latest
        latest_tag = f"{username}/{repo_name}:latest"
        run_command(f"docker tag {image_name} {latest_tag}")
        logger.info(f"✅ Tagged as latest: {latest_tag}")
        
        return image_name, latest_tag
        
    except Exception as e:
        logger.error(f"❌ Build failed: {e}")
        return None, None

def test_image_locally(image_name):
    """Test the Docker image locally."""
    logger.info("🧪 Testing Docker image locally...")
    
    try:
        # Run container
        container_name = "fraud-test-hub"
        run_command(f"docker run -d --name {container_name} -p 5000:5000 {image_name}")
        
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
            run_command(f"docker stop {container_name}", check=False)
            run_command(f"docker rm {container_name}", check=False)
            
    except Exception as e:
        logger.error(f"❌ Local test failed: {e}")
        return False

def push_to_docker_hub(image_name, latest_tag):
    """Push the image to Docker Hub."""
    logger.info("🚀 Pushing to Docker Hub...")
    
    try:
        # Push both tags
        run_command(f"docker push {image_name}")
        run_command(f"docker push {latest_tag}")
        
        logger.info("✅ Successfully pushed to Docker Hub!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Push failed: {e}")
        return False

def create_deployment_instructions(image_name):
    """Create deployment instructions."""
    logger.info("📝 Creating deployment instructions...")
    
    instructions = f"""
# Docker Hub Deployment Instructions

## 🎉 Your fraud detection model is now available on Docker Hub!

### Image Details
- **Image**: {image_name}
- **Latest**: {image_name.replace(':latest', ':latest')}

### How to Deploy

#### Option 1: Quick Deploy (Anywhere)
```bash
docker run -d --name fraud-detection -p 5000:5000 {image_name}
```

#### Option 2: With Environment Variables
```bash
docker run -d --name fraud-detection \\
  -p 5000:5000 \\
  -e LOG_LEVEL=INFO \\
  {image_name}
```

#### Option 3: Production Deployment
```bash
docker run -d --name fraud-detection \\
  -p 5000:5000 \\
  --restart unless-stopped \\
  --memory=2g \\
  --cpus=1.0 \\
  {image_name}
```

### Access Your Application
- **Web Interface**: http://localhost:5000
- **Health Check**: http://localhost:5000/health
- **API Documentation**: http://localhost:5000/docs

### Docker Compose (Recommended for Production)
Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  fraud-detection:
    image: {image_name}
    ports:
      - "5000:5000"
    restart: unless-stopped
    environment:
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Then run:
```bash
docker-compose up -d
```

### Cloud Deployment Examples

#### Digital Ocean
```bash
# SSH into your droplet
ssh root@your-droplet-ip

# Pull and run
docker pull {image_name}
docker run -d --name fraud-detection -p 80:5000 {image_name}
```

#### AWS EC2
```bash
# Install Docker on EC2
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Pull and run
docker pull {image_name}
docker run -d --name fraud-detection -p 80:5000 {image_name}
```

#### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy fraud-detection \\
  --image {image_name} \\
  --platform managed \\
  --region us-central1 \\
  --allow-unauthenticated \\
  --port 5000
```

### Monitoring and Logs
```bash
# View logs
docker logs fraud-detection

# Monitor resource usage
docker stats fraud-detection

# Access container shell
docker exec -it fraud-detection /bin/bash
```

### Updating the Application
```bash
# Pull latest version
docker pull {image_name}

# Stop and remove old container
docker stop fraud-detection
docker rm fraud-detection

# Run new version
docker run -d --name fraud-detection -p 5000:5000 {image_name}
```

## 🎯 Benefits of Docker Hub Deployment

1. **Easy Distribution**: Anyone can deploy your model with one command
2. **Version Control**: Tagged releases for different versions
3. **Scalability**: Easy to deploy to multiple environments
4. **Consistency**: Same environment everywhere
5. **Professional**: Industry-standard deployment method

## 📊 Next Steps

1. **Share the image**: Send the image name to your team
2. **Set up CI/CD**: Automate builds on code changes
3. **Monitor usage**: Track Docker Hub pulls
4. **Update regularly**: Push new versions as you improve the model
"""
    
    with open("DOCKER_HUB_DEPLOYMENT.md", "w") as f:
        f.write(instructions)
    
    logger.info("✅ Deployment instructions created: DOCKER_HUB_DEPLOYMENT.md")

def main():
    """Main deployment function."""
    logger.info("🚀 Starting Docker Hub Deployment for Fraud Detection Model")
    logger.info("=" * 70)
    
    try:
        # Step 1: Check prerequisites
        if not check_prerequisites():
            logger.error("❌ Prerequisites check failed")
            return False
        
        # Step 2: Create configuration
        if not create_docker_hub_config():
            logger.error("❌ Please update docker_hub_config.json with your Docker Hub username")
            logger.error("❌ Then run this script again")
            return False
        
        # Step 3: Login to Docker Hub
        if not login_to_docker_hub():
            logger.error("❌ Docker Hub login failed")
            return False
        
        # Step 4: Build image
        image_name, latest_tag = build_image()
        if not image_name:
            logger.error("❌ Image build failed")
            return False
        
        # Step 5: Test locally
        if not test_image_locally(image_name):
            logger.error("❌ Local test failed")
            return False
        
        # Step 6: Push to Docker Hub
        if not push_to_docker_hub(image_name, latest_tag):
            logger.error("❌ Push to Docker Hub failed")
            return False
        
        # Step 7: Create instructions
        create_deployment_instructions(image_name)
        
        # Success!
        logger.info("=" * 70)
        logger.info("🎉 DOCKER HUB DEPLOYMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"🌐 Your fraud detection model is now available on Docker Hub:")
        logger.info(f"   {image_name}")
        logger.info("")
        logger.info("📖 Deployment instructions saved to: DOCKER_HUB_DEPLOYMENT.md")
        logger.info("")
        logger.info("🚀 Next steps:")
        logger.info("   1. Share the image name with your team")
        logger.info("   2. Deploy to any environment with: docker run -p 5000:5000 " + image_name)
        logger.info("   3. Set up automated builds for future updates")
        
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