# Fraud Detection Model Deployment

This project provides two deployment modules for the fraud detection model:

## Local Deployment (`deploy_local.py`)

For testing and local inference.

### Features:
- Builds Docker image for local testing
- Starts local server on specified port (default: 5000)
- Tests model inference with sample data
- Provides web interface and API endpoints

### Usage:
```bash
# Start local deployment
python deploy_local.py

# Use custom port
python deploy_local.py --port 8080

# Stop local server
python deploy_local.py --stop
```

### Access Points:
- Web Interface: `http://localhost:5000`
- API Documentation: `http://localhost:5000/docs`
- Health Check: `http://localhost:5000/health`

## Production Deployment (`deploy_pipeline.py`)

For production deployment to Digital Ocean droplet.

### Features:
- Builds production Docker image
- Tests image locally before deployment
- Transfers image to Digital Ocean droplet
- Deploys with Docker Compose
- Verifies production deployment

### Prerequisites:
- Docker installed and running
- SSH access to Digital Ocean droplet
- Updated `deployment_config.json` with your droplet details

### Configuration:
Edit `deployment_config.json`:
```json
{
  "droplet_ip": "your-droplet-ip",
  "ssh_user": "root",
  "docker_tag": "production"
}
```

### Usage:
```bash
# Deploy to production
python deploy_pipeline.py

# Use custom config file
python deploy_pipeline.py --config my_config.json

# Use custom Docker tag
python deploy_pipeline.py --docker-tag v1.0.0
```

### Production Access:
- Web Interface: `http://your-droplet-ip`
- API Documentation: `http://your-droplet-ip/docs`
- Health Check: `http://your-droplet-ip/health`

## API Endpoints

Both deployments provide the same API endpoints:

- `GET /` - Web interface
- `GET /health` - Health check
- `GET /test-data` - Get sample test data
- `POST /predict` - Single transaction prediction
- `POST /predict-batch` - Batch transaction predictions
- `GET /model-info` - Model information

## Deployment Workflow

1. **Local Testing**: Use `deploy_local.py` to test locally
2. **Production Deployment**: Use `deploy_pipeline.py` when ready for production
3. **Verification**: Both scripts verify deployment success

## Docker

The deployment uses a single `Dockerfile` that:
- Uses Python 3.9 slim image
- Installs required dependencies
- Copies application code
- Exposes port 5000
- Runs the FastAPI application

## Security Notes

- Production deployment uses SSH key authentication
- Docker containers run with limited privileges
- Health checks ensure service availability
- Automatic restart on failure 