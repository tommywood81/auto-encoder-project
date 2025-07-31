#!/bin/bash

# FraudGuard Pro - Deployment Script
# This script automates the deployment of the fraud detection dashboard

set -e  # Exit on any error

echo "🚀 Starting FraudGuard Pro Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check if required files exist
check_files() {
    print_status "Checking required files..."
    
    required_files=(
        "app.py"
        "Dockerfile"
        "docker-compose.yml"
        "requirements.txt"
        "models/fraud_autoencoder.keras"
        "configs/final_optimized_config.yaml"
        "data/cleaned/creditcard_cleaned.csv"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "Required file not found: $file"
            exit 1
        fi
    done
    
    print_success "All required files are present"
}

# Build and start the application
deploy_app() {
    print_status "Building and starting FraudGuard Pro..."
    
    # Stop any existing containers
    print_status "Stopping existing containers..."
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Build the Docker image
    print_status "Building Docker image..."
    docker-compose build --no-cache
    
    # Start the services
    print_status "Starting services..."
    docker-compose up -d
    
    print_success "Deployment completed successfully!"
}

# Wait for application to be ready
wait_for_app() {
    print_status "Waiting for application to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/api/health &>/dev/null; then
            print_success "Application is ready!"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts - Application not ready yet..."
        sleep 2
        ((attempt++))
    done
    
    print_error "Application failed to start within expected time"
    return 1
}

# Display deployment information
show_info() {
    print_success "FraudGuard Pro is now running!"
    echo ""
    echo "📊 Dashboard URLs:"
    echo "   Main Dashboard: http://localhost:8000"
    echo "   API Documentation: http://localhost:8000/api/docs"
    echo "   Health Check: http://localhost:8000/api/health"
    echo ""
    echo "🔧 Management Commands:"
    echo "   View logs: docker-compose logs -f fraudguard-pro"
    echo "   Stop services: docker-compose down"
    echo "   Restart services: docker-compose restart"
    echo ""
    echo "📈 Features Available:"
    echo "   ✅ Real-time transaction analysis"
    echo "   ✅ Sample transaction testing"
    echo "   ✅ Batch analysis capabilities"
    echo "   ✅ Professional fraud detection UI"
    echo ""
}

# Main deployment process
main() {
    echo "🛡️  FraudGuard Pro - AI Fraud Detection System"
    echo "================================================"
    echo ""
    
    check_docker
    check_files
    deploy_app
    
    if wait_for_app; then
        show_info
    else
        print_error "Deployment failed. Check logs with: docker-compose logs fraudguard-pro"
        exit 1
    fi
}

# Run main function
main "$@" 