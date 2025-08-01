# Fraud Detection Dashboard - Docker Deployment

A simplified fraud detection dashboard that shows all transactions with fraud flags, includes a threshold slider, and displays metrics.

## Features

- **Transaction Display**: Shows all transactions with their fraud detection results
- **Adjustable Threshold**: Slider to adjust fraud detection sensitivity (0.0 - 1.0)
- **Metrics Dashboard**: Real-time metrics including fraud rate, total transactions, and processing time
- **Predict All Button**: Single button to analyze all transactions with current threshold
- **Modern UI**: Clean, responsive interface with Bootstrap styling

## Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available for the container
- Port 8000 available on your system

## Quick Start

### Option 1: Using PowerShell (Windows)
```powershell
.\deploy.ps1
```

### Option 2: Using Docker Compose directly
```bash
# Build and start the container
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Option 3: Manual Docker commands
```bash
# Build the image
docker build -t fraud-dashboard .

# Run the container
docker run -d -p 8000:8000 --name fraud-dashboard fraud-dashboard
```

## Accessing the Dashboard

Once deployed, access the dashboard at:
- **Main Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/api/health

## Usage

1. **Adjust Threshold**: Use the slider to set the fraud detection threshold
   - Lower values (0.0-0.3): More sensitive, detects more potential fraud
   - Higher values (0.7-1.0): Less sensitive, fewer false positives

2. **Predict All**: Click the "Predict All Transactions" button to analyze all transactions with the current threshold

3. **View Results**: 
   - Metrics are displayed in the top-right card
   - All transactions are shown in the table below with fraud flags
   - Fraudulent transactions are highlighted in red
   - Normal transactions are highlighted in green

## API Endpoints

- `POST /api/predict/all` - Predict fraud for all transactions with given threshold
- `GET /api/health` - Health check endpoint

## Configuration

The application uses the following files:
- `models/fraud_autoencoder.keras` - Trained fraud detection model
- `configs/final_optimized_config.yaml` - Model configuration
- `data/cleaned/creditcard_cleaned.csv` - Transaction data

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs

# Check if required files exist
ls -la models/fraud_autoencoder.keras
ls -la configs/final_optimized_config.yaml
ls -la data/cleaned/creditcard_cleaned.csv
```

### Port already in use
```bash
# Stop existing containers
docker-compose down

# Or change the port in docker-compose.yml
# ports:
#   - "8001:8000"  # Use port 8001 instead
```

### Memory issues
```bash
# Increase Docker memory limit in Docker Desktop settings
# Recommended: 4GB minimum, 8GB preferred
```

## Development

To run in development mode:
```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python app.py
```

## Performance Notes

- The application loads 1000 sample transactions for performance
- Processing time depends on the number of transactions and threshold
- Higher thresholds generally result in faster processing
- The model is loaded once at startup for optimal performance

## Security

- The container runs on port 8000
- No authentication is implemented (for demo purposes)
- Consider adding authentication for production use
- The application is designed for internal/demo use only 