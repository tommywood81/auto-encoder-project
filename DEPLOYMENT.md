# Production Deployment Guide

This guide walks you through deploying the Fraud Detection Dashboard to a production droplet using Docker and Docker Hub.

## Prerequisites

Before deploying, ensure you have:

### 1. Docker Hub Account
- Create a Docker Hub account at [hub.docker.com](https://hub.docker.com)
- Log in locally: `docker login`

### 2. Droplet Setup
Your droplet should have:
- **Docker installed**: `curl -fsSL https://get.docker.com | sh`
- **SSH access enabled**: SSH key configured
- **Open ports**: 80 (HTTP) and/or 443 (HTTPS)
- **Root access**: or sudo privileges

### 3. Local Environment
- Docker installed and running
- SSH key configured for droplet access
- Python 3.7+ with required packages

## Quick Deployment

### 1. Configure Deployment Settings

Edit `configs/deployment_config.yaml`:

```yaml
docker:
  image_name: "fraud-detection-dashboard"
  username: "your-dockerhub-username"  # Your Docker Hub username
  tag: "latest"

droplet:
  ip: "123.456.789.012"  # Your droplet's IP address
  ssh_user: "root"  # Usually "root" for DigitalOcean
  app_port: "80"  # Port to expose (80 for HTTP)
  container_name: "fraud-dashboard"
```

### 2. Validate Configuration

Test your configuration without deploying:

```bash
python deploy_droplet.py --dry-run
```

This will check:
- Docker is running
- SSH connectivity to droplet
- Docker is available on droplet
- Configuration is valid

### 3. Deploy to Production

```bash
python deploy_droplet.py
```

The script will:
1. Build and tag the Docker image
2. Push to Docker Hub
3. Connect to your droplet
4. Stop/remove existing containers
5. Pull the new image
6. Run the updated container
7. Verify deployment health

## Detailed Deployment Process

### Step 1: Build and Tag Image

The script builds your application into a Docker image:

```bash
docker build -t yourusername/fraud-detection-dashboard:latest .
```

### Step 2: Push to Docker Hub

The image is pushed to Docker Hub for distribution:

```bash
docker push yourusername/fraud-detection-dashboard:latest
```

### Step 3: Deploy to Droplet

The script connects to your droplet and:

1. **Stops existing container** (if running)
2. **Removes old container** (if exists)
3. **Pulls new image** from Docker Hub
4. **Runs new container** with proper settings

### Step 4: Health Verification

The script verifies deployment by:
- Checking container status
- Testing health endpoint
- Confirming application accessibility

## Container Configuration

The deployment uses these container settings:

```bash
docker run -d \
  --name fraud-dashboard \
  --restart unless-stopped \
  -p 80:8000 \
  -v /var/log/fraud-dashboard:/app/logs \
  -e PYTHONPATH=/app \
  -e PYTHONUNBUFFERED=1 \
  yourusername/fraud-detection-dashboard:latest
```

**Key Settings:**
- `--restart unless-stopped`: Auto-restart on failure
- `-p 80:8000`: Map port 80 to container port 8000
- `-v /var/log/fraud-dashboard:/app/logs`: Persistent log storage
- Environment variables for proper Python execution

## Monitoring and Maintenance

### View Logs

```bash
# SSH to droplet
ssh root@your-droplet-ip

# View container logs
docker logs fraud-dashboard

# Follow logs in real-time
docker logs -f fraud-dashboard
```

### Container Management

```bash
# Check container status
docker ps

# Stop container
docker stop fraud-dashboard

# Start container
docker start fraud-dashboard

# Restart container
docker restart fraud-dashboard

# Remove container
docker rm fraud-dashboard
```

### Health Monitoring

The application provides health endpoints:

```bash
# Health check
curl http://your-droplet-ip/api/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "data_loaded": true,
  "anomaly_scores_loaded": true,
  "ground_truth_loaded": true
}
```

## Troubleshooting

### Common Issues

#### 1. SSH Connection Failed
```
Error: Cannot connect to droplet via SSH
```

**Solutions:**
- Verify droplet IP is correct
- Check SSH key is properly configured
- Ensure droplet is running and accessible
- Test SSH manually: `ssh root@your-droplet-ip`

#### 2. Docker Hub Push Failed
```
Error: Failed to push image to Docker Hub
```

**Solutions:**
- Ensure you're logged in: `docker login`
- Check Docker Hub username is correct
- Verify image name follows Docker Hub naming conventions

#### 3. Container Won't Start
```
Error: Deployment verification failed
```

**Solutions:**
- Check container logs: `docker logs fraud-dashboard`
- Verify port 80 is not in use by another service
- Check droplet has sufficient resources (2GB RAM recommended)

#### 4. Health Check Fails
```
Error: Health check failed
```

**Solutions:**
- Wait longer for application startup (can take 1-2 minutes)
- Check application logs for errors
- Verify model files are properly loaded

### Debug Mode

For detailed debugging, run with verbose logging:

```bash
# Set debug logging
export PYTHONPATH=/app
python deploy_droplet.py --config configs/deployment_config.yaml
```

## Security Considerations

### Production Security

1. **Use HTTPS**: Configure SSL certificates for production
2. **Firewall**: Restrict access to necessary ports only
3. **SSH Security**: Use key-based authentication, disable password login
4. **Regular Updates**: Keep Docker and system packages updated

### SSL Configuration

For HTTPS deployment:

1. **Obtain SSL Certificate** (Let's Encrypt recommended)
2. **Update Configuration**:

```yaml
droplet:
  app_port: "443"  # Use HTTPS port

ssl:
  enabled: true
  certificate_path: "/etc/letsencrypt/live/yourdomain.com/fullchain.pem"
  private_key_path: "/etc/letsencrypt/live/yourdomain.com/privkey.pem"
```

3. **Configure Nginx** (recommended for SSL termination)

## Scaling and Performance

### Resource Requirements

**Minimum Requirements:**
- **CPU**: 1 core
- **RAM**: 2GB
- **Storage**: 10GB
- **Network**: 1Mbps

**Recommended for Production:**
- **CPU**: 2+ cores
- **RAM**: 4GB+
- **Storage**: 20GB+
- **Network**: 10Mbps+

### Performance Optimization

1. **Use SSD Storage**: Faster model loading and inference
2. **Increase Memory**: More RAM for better model performance
3. **Load Balancing**: Use multiple containers behind a load balancer
4. **CDN**: Use CDN for static assets (if applicable)

## Backup and Recovery

### Data Backup

```bash
# Backup container data
docker run --rm -v fraud-dashboard:/data -v $(pwd):/backup alpine tar czf /backup/fraud-dashboard-backup.tar.gz -C /data .

# Restore container data
docker run --rm -v fraud-dashboard:/data -v $(pwd):/backup alpine tar xzf /backup/fraud-dashboard-backup.tar.gz -C /data
```

### Disaster Recovery

1. **Regular Backups**: Automate backup of model files and data
2. **Configuration Backup**: Version control your deployment config
3. **Documentation**: Keep deployment procedures documented
4. **Testing**: Regularly test recovery procedures

## Advanced Configuration

### Custom Environment Variables

Add to `deployment_config.yaml`:

```yaml
advanced:
  environment:
    PYTHONPATH: "/app"
    PYTHONUNBUFFERED: "1"
    DEBUG: "false"
    LOG_LEVEL: "INFO"
    MODEL_PATH: "/app/models/fraud_autoencoder.keras"
```

### Volume Mounts

Mount additional directories:

```yaml
advanced:
  volumes:
    logs: "/var/log/fraud-dashboard:/app/logs"
    data: "/var/fraud-dashboard/data:/app/data"
    models: "/var/fraud-dashboard/models:/app/models"
```

### Resource Limits

Set container resource limits:

```yaml
advanced:
  memory_limit: "4g"
  cpu_limit: "2.0"
```

## Support and Maintenance

### Regular Maintenance

1. **Weekly**: Check container logs and health
2. **Monthly**: Update Docker images and dependencies
3. **Quarterly**: Review and update security configurations
4. **Annually**: Plan for major version upgrades

### Monitoring Setup

Consider setting up monitoring for:
- Container health and restart events
- Application performance metrics
- System resource usage
- Error rates and response times

---

## Quick Reference

### Essential Commands

```bash
# Deploy to production
python deploy_droplet.py

# Validate configuration
python deploy_droplet.py --dry-run

# View logs
ssh root@your-droplet-ip "docker logs fraud-dashboard"

# Restart application
ssh root@your-droplet-ip "docker restart fraud-dashboard"

# Check health
curl http://your-droplet-ip/api/health
```

### Configuration File

`configs/deployment_config.yaml` - Main deployment configuration
`docker-compose.yml` - Local development configuration
`Dockerfile` - Container build instructions

### Important URLs

- **Dashboard**: `http://your-droplet-ip`
- **Health Check**: `http://your-droplet-ip/api/health`
- **Docker Hub**: `https://hub.docker.com/r/yourusername/fraud-detection-dashboard`

---

*This deployment guide ensures your fraud detection dashboard is production-ready with proper monitoring, security, and maintenance procedures.* 