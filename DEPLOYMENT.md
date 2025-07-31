# FraudGuard Pro - Deployment Guide

## 🚀 Quick Start

### Local Development

1. **Activate Virtual Environment**
   ```bash
   # Windows
   env\Scripts\activate
   
   # Linux/Mac
   source env/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Access the Dashboard**
   - Main Dashboard: http://localhost:8000
   - API Documentation: http://localhost:8000/api/docs
   - Health Check: http://localhost:8000/api/health

### Docker Deployment

1. **Build and Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Or use the deployment script (Linux/Mac)**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Access the Dashboard**
   - Main Dashboard: http://localhost:8000
   - API Documentation: http://localhost:8000/api/docs

## 📊 Dashboard Features

### Real-time Transaction Analysis
- Input transaction details (amount, customer age, payment method, etc.)
- Get instant fraud probability and risk assessment
- View confidence scores and anomaly detection results

### Sample Transaction Testing
- Click on pre-loaded sample transactions (both fraudulent and normal)
- Test the system with real data examples
- See how the model performs on different transaction types

### Batch Analysis
- Analyze multiple transactions at once
- Get comprehensive results for all samples
- View processing times and performance metrics

## 🔧 API Endpoints

### Health Check
```bash
GET /api/health
```

### System Status
```bash
GET /api/status
```

### Get Sample Transactions
```bash
GET /api/samples
```

### Single Transaction Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "amount": 150.50,
  "customer_age": 35,
  "payment_method": "credit_card",
  "merchant_category": "electronics",
  "transaction_time": "2024-01-15T14:30:00",
  "location": "New York, NY"
}
```

### Batch Transaction Prediction
```bash
POST /api/predict/batch
Content-Type: application/json

{
  "transactions": [
    {
      "amount": 150.50,
      "customer_age": 35,
      "payment_method": "credit_card",
      "merchant_category": "electronics",
      "transaction_time": "2024-01-15T14:30:00",
      "location": "New York, NY"
    }
  ]
}
```

## 🐳 Docker Configuration

### Services
- **fraudguard-pro**: Main FastAPI application
- **redis**: Optional caching layer
- **nginx**: Optional load balancer and SSL termination

### Environment Variables
- `PYTHONPATH`: Python path configuration
- `MODEL_PATH`: Path to the trained model
- `CONFIG_PATH`: Path to configuration file

### Volumes
- `./models`: Read-only access to trained models
- `./data`: Read-only access to data files
- `./configs`: Read-only access to configuration files
- `./logs`: Application logs
- `./predictions`: Prediction results

## 🔍 Monitoring

### Health Checks
The application includes built-in health checks:
- Application health: `/api/health`
- System status: `/api/status`
- Docker health check: Configured in Dockerfile

### Logs
```bash
# View application logs
docker-compose logs -f fraudguard-pro

# View all service logs
docker-compose logs -f
```

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure `models/fraud_autoencoder.keras` exists
   - Check `configs/final_optimized_config.yaml` is valid
   - Verify `data/cleaned/creditcard_cleaned.csv` is present

2. **Port Conflicts**
   - Change port in `docker-compose.yml` if 8000 is in use
   - Check for other services using the same port

3. **Memory Issues**
   - Increase Docker memory allocation
   - Consider using GPU if available

### Performance Optimization

1. **Enable GPU Support**
   ```bash
   # Uncomment in requirements.txt
   # tensorflow-gpu>=2.8.0
   ```

2. **Use Redis for Caching**
   - Enable Redis service in docker-compose.yml
   - Configure caching in the application

3. **Load Balancing**
   - Use nginx service for load balancing
   - Configure multiple application instances

## 📈 Production Deployment

### Security Considerations
- Use HTTPS in production
- Implement authentication and authorization
- Secure API endpoints
- Use environment variables for sensitive data

### Scaling
- Use multiple application instances
- Implement load balancing
- Use Redis for session management
- Consider using a CDN for static assets

### Monitoring
- Implement logging aggregation
- Set up metrics collection
- Configure alerting
- Use APM tools for performance monitoring

## 🎯 Next Steps

1. **Model Updates**
   - Retrain model with new data
   - Update configuration files
   - Deploy new model version

2. **Feature Enhancements**
   - Add user authentication
   - Implement audit logging
   - Add more transaction types
   - Enhance visualization features

3. **Integration**
   - Connect to real transaction systems
   - Implement webhook notifications
   - Add database persistence
   - Integrate with monitoring tools 