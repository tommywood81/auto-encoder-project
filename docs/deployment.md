# Deployment Documentation

## Deployment Options

### Local Development
```bash
python app.py
# Access: http://localhost:8000
```

### Docker Deployment
```bash
python deploy_local.py
# or
docker-compose up -d
```

### Cloud Deployment with DigitalOcean

The project includes a sophisticated deployment script for DigitalOcean droplets:

```python
python deploy_droplet.py --config configs/deployment_config.yaml
```

## DigitalOcean Deployment Process

The `deploy_droplet.py` script provides production-grade deployment:

**Key Features:**
- Automated deployment pipeline
- Health verification
- Zero-downtime deployment
- Comprehensive error handling

**Deployment Steps:**

1. **Prerequisites Check**
```python
def check_prerequisites(self) -> bool:
    """Check if all prerequisites are met."""
    # Check Docker installation
    self.run_command("docker info", check=True, capture_output=True)
    
    # Check SSH connectivity
    ssh_test_cmd = f"ssh {self.ssh_user}@{self.droplet_ip} whoami"
    result = subprocess.run(ssh_test_cmd, shell=True, capture_output=True, timeout=30)
```

2. **Build & Push**
```python
def build_and_tag_image(self):
    """Build and tag the Docker image."""
    build_cmd = f"docker build -t {self.full_image_name} ."
    self.run_command(build_cmd, check=True)

def push_to_docker_hub(self):
    """Push the image to Docker Hub."""
    push_cmd = f"docker push {self.full_image_name}"
    self.run_command(push_cmd, check=True)
```

3. **Deploy to Droplet**
```python
def deploy_to_droplet(self):
    """Deploy the application to the droplet."""
    # Stop and remove existing container
    stop_cmd = f"ssh {self.ssh_user}@{self.droplet_ip} docker stop {self.container_name}"
    
    # Pull and run new container
    run_cmd = f"ssh {self.ssh_user}@{self.droplet_ip} docker run -d \
        --name {self.container_name} \
        --restart unless-stopped \
        -p {self.app_port}:8000 \
        -v /var/log/fraud-dashboard:/app/logs \
        {self.full_image_name}"
```

4. **Health Verification**
```python
def verify_deployment(self):
    """Verify the deployment is healthy."""
    # Check container status
    status_cmd = f"docker ps --filter name={self.container_name}"
    
    # Test health endpoint
    health_cmd = f"curl -f http://localhost:{self.app_port}/api/health"
```

## Configuration

```yaml
# configs/deployment_config.yaml
docker:
  image_name: fraud-detection-dashboard
  username: your-dockerhub-username
  tag: latest

droplet:
  ip: your-droplet-ip
  ssh_user: root
  app_port: 80
  container_name: fraud-dashboard
```
