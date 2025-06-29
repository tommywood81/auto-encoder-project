#!/usr/bin/env python3
"""
Simple Docker Hub deployment - just build and push.
"""

import subprocess
import os

def run_command(command):
    """Run a command and return result."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e.stderr}")
        return False

def main():
    print("🚀 Simple Docker Hub Deployment")
    print("=" * 40)
    
    # Use simple Dockerfile
    if os.path.exists("Dockerfile.simple"):
        run_command("copy Dockerfile.simple Dockerfile")
    
    # Build image
    image_name = "tommyboy777/fraud-detection-api:latest"
    print(f"Building {image_name}...")
    
    if not run_command(f"docker build -t {image_name} ."):
        print("❌ Build failed")
        return
    
    # Push to Docker Hub
    print("Pushing to Docker Hub...")
    if run_command(f"docker push {image_name}"):
        print("🎉 Successfully deployed to Docker Hub!")
        print(f"Image: {image_name}")
        print("Deploy with: docker run -p 5000:5000 " + image_name)
    else:
        print("❌ Failed to push to Docker Hub")

if __name__ == "__main__":
    main() 