#!/usr/bin/env python3
"""
Quick Docker Hub deployment with optimized image.
"""

import subprocess
import os
import sys

def run_command(command):
    """Run a command and return result."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Quick Docker Hub Deployment")
    print("=" * 50)
    
    # Use optimized Dockerfile
    dockerfile = "Dockerfile.optimized"
    dockerignore = ".dockerignore.optimized"
    
    # Copy optimized files
    if os.path.exists(dockerfile):
        run_command(f"copy {dockerfile} Dockerfile")
    if os.path.exists(dockerignore):
        run_command(f"copy {dockerignore} .dockerignore")
    
    # Build optimized image
    image_name = "tommyboy777/fraud-detection-api:latest"
    if not run_command(f"docker build -t {image_name} ."):
        return False
    
    # Test locally
    print("🧪 Testing locally...")
    container_name = "fraud-test-quick"
    run_command(f"docker run -d --name {container_name} -p 5000:5000 {image_name}")
    
    # Wait a bit
    import time
    time.sleep(10)
    
    # Check if it's running
    result = subprocess.run("docker ps", shell=True, capture_output=True, text=True)
    if container_name in result.stdout:
        print("✅ Container is running!")
    else:
        print("❌ Container failed to start")
        run_command(f"docker logs {container_name}")
    
    # Cleanup
    run_command(f"docker stop {container_name}")
    run_command(f"docker rm {container_name}")
    
    # Push to Docker Hub
    print("🚀 Pushing to Docker Hub...")
    if run_command(f"docker push {image_name}"):
        print("🎉 Successfully deployed to Docker Hub!")
        print(f"Image: {image_name}")
        print("Deploy with: docker run -p 5000:5000 " + image_name)
    else:
        print("❌ Failed to push to Docker Hub")

if __name__ == "__main__":
    main() 