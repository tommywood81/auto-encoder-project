# 🚀 Fraud Detection Model Deployment Guide

This guide explains how to deploy the trained fraud detection model to your Digital Ocean droplet using Docker.

## 📋 Prerequisites

Before deploying, ensure you have:

1. **Docker** installed and running on your local machine
2. **SSH access** to your Digital Ocean droplet (SSH key configured)
3. **Python 3.8+** with required dependencies
4. **Trained model** files in the `models/` directory

## 🏗️ Architecture Overview

The deployment consists of:

- **Flask API Server** (`app.py`) - Serves the model via REST API
- **Web Interface** (`templates/index.html`) - User-friendly interface for testing
- **Docker Container** - Production-ready containerization
- **Digital Ocean Droplet** - Ubuntu 22.04 server for hosting

## 📁 Project Structure

```
auto-encoder-project/
├── app.py                    # Flask API server
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
├── deploy_local.py          # Local deployment script
├── deployment_config.json   # Deployment configuration
├── templates/
│   └── index.html          # Web interface
├── models/                  # Trained model files
├── data/                    # Data files
└── src/                     # Source code
```

## 🔧 Configuration

### 1. Create Deployment Configuration

The deployment script will create a default configuration file if it doesn't exist:

```json
{
  "droplet_ip": "209.38.89.159",
  "ssh_user": "root",
  "docker_tag": "latest"
}
```

### 2. Update Configuration

Edit `deployment_config.json` with your specific settings:

```json
{
  "droplet_ip": "YOUR_DROPLET_IP",
  "ssh_user": "root",
  "docker_tag": "v1.0.0"
}
```

## 🚀 Deployment Steps

### Option 1: Automated Deployment (Recommended)

1. **Run the deployment script:**

```bash
python deploy_local.py
```

This script will:
- ✅ Check prerequisites (Docker, SSH access)
- 🏗️ Build Docker image locally
- 🧪 Test the image locally
- 📦 Save and transfer image to droplet
- 🚀 Deploy and start the application

### Option 2: Manual Deployment

If you prefer manual deployment:

1. **Build Docker image:**
```bash
docker build -t fraud-detection-api:latest .
```

2. **Test locally:**
```bash
docker run -d -p 5000:5000 fraud-detection-api:latest
curl http://localhost:5000/health
```

3. **Save and transfer image:**
```bash
docker save fraud-detection-api:latest -o fraud-detection-api.tar
gzip fraud-detection-api.tar
scp fraud-detection-api.tar.gz root@209.38.89.159:/tmp/
```

4. **Deploy to droplet:**
```bash
ssh root@209.38.89.159
cd /tmp
docker load -i fraud-detection-api.tar.gz
# Follow the deployment script steps manually
```

## 🌐 Accessing the Application

After successful deployment, access the application at:

- **Web Interface**: `http://209.38.89.159`
- **API Health Check**: `http://209.38.89.159/health`
- **API Documentation**: See API endpoints below

## 📊 API Endpoints

### Health Check
```bash
GET /health
```
Returns application status and model loading status.

### Get Test Data
```bash
GET /test-data
```
Returns 20 test data points for inference.

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "features": {
    "transaction_amount": 100.0,
    "customer_age": 25,
    "quantity": 1,
    ...
  }
}
```

### Batch Prediction
```bash
POST /predict-batch
Content-Type: application/json

{
  "transactions": [
    {
      "features": {...}
    },
    {
      "features": {...}
    }
  ]
}
```

### Model Information
```bash
GET /model-info
```
Returns information about the loaded model.

## 🔍 Using the Web Interface

1. **Open** `http://209.38.89.159` in your browser
2. **Select** a test data point from the grid
3. **Click** "Analyze Selected Transaction"
4. **View** the fraud detection results

The interface shows:
- 🛡️ Fraud/Legitimate prediction
- 📊 Confidence score
- 📈 Anomaly score vs threshold
- 📋 Model information

## 🐳 Docker Commands

### View running containers:
```bash
docker ps
```

### View logs:
```bash
docker logs fraud-detection-api
```

### Stop the application:
```bash
docker stop fraud-detection-api
```

### Restart the application:
```bash
docker restart fraud-detection-api
```

## 🔧 Troubleshooting

### Common Issues

1. **SSH Connection Failed**
   - Ensure SSH key is configured
   - Check droplet IP address
   - Verify SSH user permissions

2. **Docker Build Failed**
   - Check Docker daemon is running
   - Verify all files are present
   - Check Dockerfile syntax

3. **Model Loading Failed**
   - Ensure model files exist in `models/` directory
   - Check model file permissions
   - Verify TensorFlow compatibility

4. **Application Health Check Failed**
   - Check container logs: `docker logs fraud-detection-api`
   - Verify port 80 is open on droplet
   - Check firewall settings

### Debug Commands

```bash
# Check container status
docker ps -a

# View detailed logs
docker logs -f fraud-detection-api

# Access container shell
docker exec -it fraud-detection-api /bin/bash

# Check droplet resources
ssh root@209.38.89.159 "df -h && free -h"
```

## 📈 Monitoring

### Health Monitoring
The application includes health checks that run every 30 seconds.

### Logs
Application logs are stored in `/opt/fraud-detection-api/logs/` on the droplet.

### Performance
Monitor resource usage:
```bash
# On droplet
htop
docker stats
```

## 🔄 Updating the Application

To update the deployed application:

1. **Make your changes** to the code
2. **Update the Docker tag** in configuration
3. **Run deployment again:**
```bash
python deploy_local.py --docker-tag v1.0.1
```

## 🛡️ Security Considerations

1. **Firewall**: Ensure only necessary ports are open
2. **SSH**: Use key-based authentication only
3. **Docker**: Run containers as non-root user
4. **Updates**: Keep system packages updated
5. **Monitoring**: Monitor for suspicious activity

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review application logs
3. Verify configuration settings
4. Test locally before deploying

## 🎉 Success!

Once deployed, your fraud detection model will be available at:
**http://209.38.89.159**

The application provides:
- ✅ Real-time fraud detection
- 🌐 Web interface for testing
- 📊 REST API for integration
- 🛡️ Production-ready deployment
- 📈 Health monitoring

---

*This deployment pipeline demonstrates a production-ready ML model deployment with proper containerization, monitoring, and user interface.* 