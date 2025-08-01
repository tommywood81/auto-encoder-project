# Fraud Detection Dashboard Deployment Script for Windows

Write-Host "🚀 Building and deploying Fraud Detection Dashboard..." -ForegroundColor Green

# Build the Docker image
Write-Host "📦 Building Docker image..." -ForegroundColor Yellow
docker-compose build

# Run the container
Write-Host "🏃 Starting container..." -ForegroundColor Yellow
docker-compose up -d

# Wait for the service to be ready
Write-Host "⏳ Waiting for service to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check if the service is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Service is running successfully!" -ForegroundColor Green
        Write-Host "🌐 Dashboard available at: http://localhost:8000" -ForegroundColor Cyan
        Write-Host "📊 API documentation at: http://localhost:8000/api/docs" -ForegroundColor Cyan
    }
} catch {
    Write-Host "❌ Service failed to start. Check logs with: docker-compose logs" -ForegroundColor Red
    exit 1
}

Write-Host "🎉 Deployment complete!" -ForegroundColor Green 