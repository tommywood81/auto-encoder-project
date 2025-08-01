#!/bin/bash

echo "🚀 Building and deploying Fraud Detection Dashboard..."

# Build the Docker image
echo "📦 Building Docker image..."
docker-compose build

# Run the container
echo "🏃 Starting container..."
docker-compose up -d

# Wait for the service to be ready
echo "⏳ Waiting for service to be ready..."
sleep 10

# Check if the service is running
if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✅ Service is running successfully!"
    echo "🌐 Dashboard available at: http://localhost:8000"
    echo "📊 API documentation at: http://localhost:8000/api/docs"
else
    echo "❌ Service failed to start. Check logs with: docker-compose logs"
    exit 1
fi

echo "🎉 Deployment complete!" 