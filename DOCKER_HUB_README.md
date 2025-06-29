# 🐳 Docker Hub Deployment Guide

This guide will help you deploy your fraud detection model to Docker Hub for easy distribution and deployment across any environment.

## 🎯 Why Docker Hub?

- **Easy Distribution**: Share your model with one command
- **Version Control**: Tagged releases for different versions
- **Cloud Agnostic**: Deploy anywhere Docker runs
- **Professional**: Industry-standard deployment method
- **Scalable**: Easy to deploy to multiple environments

## 🚀 Quick Start

### 1. Prerequisites

- Docker installed and running
- Docker Hub account
- Trained model files in `models/` directory

### 2. Setup Docker Hub Configuration

```bash
# Run the deployment script (it will create config for you)
python docker-hub-deploy.py
```

This will create a `docker_hub_config.json` file. Edit it with your Docker Hub username:

```json
{
  "docker_hub_username": "your-actual-username",
  "repository_name": "fraud-detection-api",
  "image_tag": "latest",
  "description": "AI-powered fraud detection using autoencoders"
}
```

### 3. Deploy to Docker Hub

```bash
# Run the deployment script
python docker-hub-deploy.py
```

The script will:
- ✅ Check prerequisites
- 🔐 Login to Docker Hub
- 🏗️ Build the Docker image
- 🧪 Test locally
- 🚀 Push to Docker Hub
- 📝 Create deployment instructions

## 📦 Manual Deployment

If you prefer to deploy manually:

### 1. Build the Image

```bash
# Build with your Docker Hub username
docker build -t your-username/fraud-detection-api:latest .

# Tag for version control
docker tag your-username/fraud-detection-api:latest your-username/fraud-detection-api:v1.0.0
```

### 2. Test Locally

```bash
# Run container
docker run -d --name fraud-test -p 5000:5000 your-username/fraud-detection-api:latest

# Test health endpoint
curl http://localhost:5000/health

# Cleanup
docker stop fraud-test && docker rm fraud-test
```

### 3. Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push images
docker push your-username/fraud-detection-api:latest
docker push your-username/fraud-detection-api:v1.0.0
```

## 🌐 Deploying Your Model

Once your image is on Docker Hub, anyone can deploy it with:

### Simple Deployment

```bash
docker run -d --name fraud-detection -p 5000:5000 your-username/fraud-detection-api:latest
```

### Production Deployment

```bash
docker run -d \
  --name fraud-detection \
  -p 5000:5000 \
  --restart unless-stopped \
  --memory=2g \
  --cpus=1.0 \
  your-username/fraud-detection-api:latest
```

### Using Docker Compose

```bash
# Update docker-compose.yml with your image
# Then run:
docker-compose up -d
```

## 🏗️ Architecture

### Multi-Stage Build

The Dockerfile uses a multi-stage build for optimization:

1. **Builder Stage**: Installs dependencies and builds the application
2. **Production Stage**: Creates a minimal runtime image

### Security Features

- ✅ Non-root user execution
- ✅ Minimal base image (python:3.11-slim)
- ✅ Health checks
- ✅ Resource limits
- ✅ No sensitive data in image

### Image Layers

```
Base Image (python:3.11-slim)
├── System Dependencies (curl)
├── Python Virtual Environment
├── Application Code
├── User Setup (appuser)
└── Health Check & CMD
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `PYTHONUNBUFFERED` | `1` | Unbuffered Python output |

### Ports

| Port | Description |
|------|-------------|
| `5000` | FastAPI application |

### Volumes

| Path | Description |
|------|-------------|
| `/app/logs` | Application logs |

## 📊 Monitoring

### Health Checks

The container includes built-in health checks:

```bash
# Check container health
docker ps

# View health check logs
docker inspect fraud-detection | grep -A 10 Health
```

### Logs

```bash
# View application logs
docker logs fraud-detection

# Follow logs in real-time
docker logs -f fraud-detection
```

### Resource Usage

```bash
# Monitor resource usage
docker stats fraud-detection
```

## 🔄 CI/CD Integration

### GitHub Actions

The repository includes a GitHub Actions workflow that automatically:

1. Builds the Docker image on code changes
2. Runs tests
3. Pushes to Docker Hub
4. Supports multiple platforms (amd64, arm64)

### Setup GitHub Secrets

Add these secrets to your GitHub repository:

- `DOCKER_HUB_USERNAME`: Your Docker Hub username
- `DOCKER_HUB_TOKEN`: Your Docker Hub access token

### Automated Deployment

```bash
# Tag a release to trigger deployment
git tag v1.0.0
git push origin v1.0.0
```

## 🌍 Cloud Deployment Examples

### Digital Ocean

```bash
# SSH into your droplet
ssh root@your-droplet-ip

# Pull and run
docker pull your-username/fraud-detection-api:latest
docker run -d --name fraud-detection -p 80:5000 your-username/fraud-detection-api:latest
```

### AWS EC2

```bash
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Deploy
docker pull your-username/fraud-detection-api:latest
docker run -d --name fraud-detection -p 80:5000 your-username/fraud-detection-api:latest
```

### Google Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy fraud-detection \
  --image your-username/fraud-detection-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 5000
```

### Azure Container Instances

```bash
# Deploy to Azure
az container create \
  --resource-group myResourceGroup \
  --name fraud-detection \
  --image your-username/fraud-detection-api:latest \
  --ports 5000 \
  --dns-name-label fraud-detection
```

## 🔄 Updating Your Model

### Version Control

```bash
# Build new version
docker build -t your-username/fraud-detection-api:v1.1.0 .

# Push new version
docker push your-username/fraud-detection-api:v1.1.0
docker push your-username/fraud-detection-api:latest

# Update running container
docker pull your-username/fraud-detection-api:latest
docker stop fraud-detection
docker rm fraud-detection
docker run -d --name fraud-detection -p 5000:5000 your-username/fraud-detection-api:latest
```

### Rolling Updates

```bash
# Using Docker Compose
docker-compose pull
docker-compose up -d
```

## 🛠️ Troubleshooting

### Common Issues

#### Image Build Fails

```bash
# Check Docker daemon
docker info

# Clean up Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t your-username/fraud-detection-api:latest .
```

#### Container Won't Start

```bash
# Check logs
docker logs fraud-detection

# Run interactively for debugging
docker run -it --rm your-username/fraud-detection-api:latest /bin/bash
```

#### Health Check Fails

```bash
# Check if application is running
docker exec fraud-detection curl http://localhost:5000/health

# Check resource usage
docker stats fraud-detection
```

### Performance Optimization

#### Memory Issues

```bash
# Increase memory limit
docker run -d --memory=4g your-username/fraud-detection-api:latest
```

#### CPU Issues

```bash
# Limit CPU usage
docker run -d --cpus=2.0 your-username/fraud-detection-api:latest
```

## 📈 Best Practices

### 1. Image Optimization

- ✅ Use multi-stage builds
- ✅ Minimize layers
- ✅ Remove unnecessary files
- ✅ Use specific base image versions

### 2. Security

- ✅ Run as non-root user
- ✅ Scan images for vulnerabilities
- ✅ Keep base images updated
- ✅ Don't include secrets in images

### 3. Monitoring

- ✅ Use health checks
- ✅ Monitor resource usage
- ✅ Set up logging
- ✅ Use proper restart policies

### 4. Versioning

- ✅ Tag releases with semantic versions
- ✅ Keep `latest` tag updated
- ✅ Document breaking changes
- ✅ Test before pushing

## 🎉 Success!

Your fraud detection model is now:

- ✅ Containerized and optimized
- ✅ Available on Docker Hub
- ✅ Ready for deployment anywhere
- ✅ Professionally packaged
- ✅ Easy to distribute and update

## 📞 Support

For issues with Docker Hub deployment:

1. Check the troubleshooting section
2. Review Docker logs
3. Verify Docker Hub credentials
4. Ensure model files are present

---

**Happy Deploying! 🚀** 