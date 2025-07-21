# Autoencoder Fraud Detection Dashboard

A production-ready, interactive dashboard demonstrating unsupervised autoencoder-based fraud detection in transaction data. This project showcases how machine learning can identify anomalous transactions without requiring labeled fraud data.

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** with virtual environment support
- **Docker** (for production deployment)
- **Git** for version control

### Local Development Setup

1. **Clone and navigate to the project:**
   ```bash
   git clone <your-repo-url>
   cd auto-encoder-project
   ```

2. **Activate the virtual environment:**
   ```bash
   # Windows
   .\env\Scripts\Activate.ps1
   
   # Linux/Mac
   source env/bin/activate
   ```

3. **Start the development server:**
   ```bash
   python app.py
   ```

4. **Access the dashboard:**
   - **Dashboard**: http://localhost:5000
   - **API Documentation**: http://localhost:5000/docs
   - **Health Check**: http://localhost:5000/health

### Production Deployment

#### Local Docker Deployment
```bash
# Deploy using Docker
python deploy_local.py

# Stop the deployment
python deploy_local.py --stop
```

#### Digital Ocean Production Deployment
```bash
# Deploy to Digital Ocean droplet
python deploy_pipeline.py
```

## 📊 Dashboard Features

### Business Use Tab
- **One-click Analysis**: Single "Analyze Transactions" button for instant results
- **Pagination**: View 100 transactions per page for optimal performance
- **Real-time Statistics**: Live fraud detection metrics
- **Feature Importance**: Random Forest explainer shows which features contributed to each decision
- **Sorted by Date**: Transactions displayed chronologically (earliest first)

### Data Science Tab
- **3D Latent Space Visualization**: Interactive Plotly visualization of the autoencoder's learned representations
- **Model Performance Metrics**: Comprehensive evaluation of the fraud detection model
- **Feature Engineering Insights**: Detailed analysis of engineered features

### Key Metrics Displayed
- **Total Transactions**: Number of transactions analyzed
- **Flagged as Fraud**: Transactions identified as potentially fraudulent
- **Actual Fraud Caught**: True positives (correctly identified frauds)
- **Precision**: Percentage of flagged transactions that were actually fraudulent

## 🔧 Technical Architecture

### Model Details
- **Type**: Unsupervised Autoencoder
- **Feature Strategy**: Combined (39 engineered features)
- **Training**: Unsupervised on clean transaction data
- **Threshold**: 95th percentile reconstruction error
- **Performance**: AUC-ROC of 0.73

### Feature Engineering
The model uses 39 engineered features including:
- **Baseline Features**: Transaction amount, quantity, customer age, account age
- **Temporal Features**: Hour of day, late night flags, time-based patterns
- **Behavioral Features**: Rolling averages, burst transaction detection
- **Demographic Features**: Age bands, location frequency
- **Fraud Flags**: High amount, new account, unusual location indicators

### API Endpoints

#### Core Endpoints
- `GET /health` - Application health check
- `GET /available-dates` - List of available transaction dates
- `GET /model-info` - Model configuration and performance details

#### Business Analysis Endpoints
- `GET /api/all-columns-transactions-fast` - Optimized transaction analysis with pagination
  - Parameters: `date` (optional), `threshold` (optional), `page` (optional)
  - Returns: Paginated transaction data with fraud predictions and feature importance

#### Data Science Endpoints
- `GET /api/latent-space-3d` - 3D latent space visualization data
- `GET /api/sample-anomaly-scores` - Sample anomaly scores for analysis

## 🛠️ Development

### Project Structure
```
auto-encoder-project/
├── app.py                 # FastAPI application
├── deploy_local.py        # Local Docker deployment
├── deploy_pipeline.py     # Production deployment
├── src/                   # Core application code
│   ├── models/           # Autoencoder model definitions
│   ├── feature_factory/  # Feature engineering strategies
│   ├── evaluation/       # Model evaluation utilities
│   └── config.py         # Configuration management
├── configs/              # YAML configuration files
├── models/               # Trained model files
├── data/                 # Dataset files
├── templates/            # HTML templates
├── static/               # Static assets
└── tests/                # Unit tests
```

### Running Tests
```bash
python run_tests.py
```

### Training New Models
```bash
# Train final model with combined features
python train_final_model.py

# Run full pipeline with different strategies
python run_pipeline.py --strategy combined
```

## 🚀 Deployment Options

### Local Development
Perfect for development, testing, and demonstrations:
- Fast startup and iteration
- Full debugging capabilities
- No external dependencies

### Docker Local Deployment
Ideal for consistent environments and testing:
- Containerized application
- Reproducible builds
- Easy cleanup and restart

### Digital Ocean Production
Production-ready deployment with:
- Scalable infrastructure
- Public accessibility
- Automated deployment pipeline
- Health monitoring

## 🔍 Troubleshooting

### Common Issues

#### Server Won't Start
```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000  # Windows
lsof -i :5000                 # Linux/Mac

# Kill process using port 5000
taskkill /F /PID <PID>        # Windows
kill -9 <PID>                 # Linux/Mac
```

#### Model Loading Errors
- Ensure `models/final_model.h5` exists
- Check that `models/final_model_scaler.pkl` is present
- Verify `models/final_model_info.yaml` contains threshold information

#### Docker Issues
```bash
# Check Docker status
docker info

# Clean up containers
docker stop $(docker ps -q)
docker rm $(docker ps -aq)

# Rebuild image
docker build -t fraud-demo:local .
```

#### SSH Connection Issues (Production)
- Ensure SSH keys are configured for the droplet
- Verify droplet IP address in `deployment_config.json`
- Check firewall settings on the droplet

### Performance Optimization

#### For Large Datasets
- The dashboard uses pagination (100 rows per page)
- Feature importance is calculated only for displayed transactions
- Consider increasing server resources for production use

#### Memory Usage
- Model loads ~500MB of data into memory
- Feature engineering requires additional memory
- Monitor system resources during heavy usage

## 📈 Model Performance

### Current Performance
- **AUC-ROC**: 0.73
- **Threshold**: 95th percentile
- **Feature Count**: 39 engineered features
- **Training Data**: 22,580 transactions

### Performance Interpretation
- **AUC-ROC 0.73**: The model can correctly rank 73% of fraud cases above non-fraud cases
- **Unsupervised Learning**: No labeled fraud data was used during training
- **Anomaly Detection**: Identifies unusual patterns rather than known fraud types

### Improvement Opportunities
1. **Feature Engineering**: Add domain-specific features
2. **Model Architecture**: Experiment with different autoencoder designs
3. **Data Quality**: Improve data preprocessing and cleaning
4. **Semi-supervised Learning**: Incorporate labeled fraud data when available

## 🤝 Contributing

### Development Workflow
1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes following PEP 8 style guidelines
3. Add tests for new functionality
4. Update documentation as needed
5. Submit a pull request

### Code Standards
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with FastAPI for high-performance API development
- Uses TensorFlow/Keras for deep learning models
- Plotly for interactive visualizations
- Pandas and NumPy for data processing
- Docker for containerization and deployment

---

**Ready to detect fraud?** Start with the [Quick Start](#-quick-start) section above!

**Questions or issues?** Check the [Troubleshooting](#-troubleshooting) section or open an issue on GitHub. 