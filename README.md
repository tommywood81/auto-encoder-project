# Fraud Detection Dashboard

A production-ready fraud detection system using unsupervised autoencoders to identify anomalous credit card transactions.

## Overview

This system learns normal transaction patterns and flags anomalies as potential fraud. Unlike supervised approaches, it requires no fraud labels for training, making it capable of detecting novel fraud patterns.

**Key Features:**
- **Unsupervised Learning** - No fraud labels required
- **Interactive Dashboard** - Real-time analysis with adjustable sensitivity  
- **Production Ready** - Docker deployment with health monitoring
- **Comprehensive Testing** - 7 test suites ensuring reliability
- **Modular Design** - Config-driven architecture

## Live Demo

🚀 **[Try the Dashboard](https://tinyurl.com/yck8p9p3)** - Live fraud detection system

## Quick Start

### Local Development
```bash
# Clone and setup
git clone <repository-url>
cd auto-encoder-project
pip install -r requirements.txt

# Run dashboard
python app.py
```
Access at `http://localhost:8000`

### Docker Deployment
```bash
# Automated local deployment
python deploy_local.py

# Or manual Docker
docker-compose up -d
```

### Cloud Deployment
```bash
# Deploy to DigitalOcean droplet
python deploy_droplet.py --config configs/deployment_config.yaml
```

## How It Works

1. **Model Training** - Autoencoder learns normal transaction patterns
2. **Feature Engineering** - 85+ domain-specific features extracted
3. **Anomaly Detection** - High reconstruction error = potential fraud
4. **Interactive Analysis** - Dashboard with adjustable sensitivity (50%-99%)

## Performance

- **AUC ROC**: 0.937+ (exceeds industry standards)
- **Processing Speed**: 1000+ transactions/second
- **Memory Usage**: ~2GB optimized for production

## Documentation

📚 **Detailed Documentation:**
- [Model Architecture](docs/model.md) - Autoencoder implementation
- [API Documentation](docs/api.md) - FastAPI endpoints
- [Feature Engineering](docs/features.md) - Domain-driven features
- [Deployment Guide](docs/deployment.md) - Production deployment

## Project Structure

```
auto-encoder-project/
├── src/                    # Core application code
├── docs/                   # Detailed documentation
├── configs/                # YAML configurations
├── tests/                  # Test suite
├── models/                 # Trained models
├── app.py                  # FastAPI dashboard
├── main.py                 # Training pipeline
├── deploy_droplet.py       # Cloud deployment
└── deploy_local.py         # Local deployment
```

## Testing

```bash
# Run all tests
python run_tests.py

# Individual tests
python run_single_test.py auc_test           # Performance validation
python run_single_test.py reproducibility_test # Consistency checks
python run_single_test.py no_data_leak       # Data integrity
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Run tests (`python run_tests.py`)
4. Commit changes (`git commit -am 'Add feature'`)
5. Push branch (`git push origin feature/new-feature`)
6. Create Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

*Built with Python, FastAPI, TensorFlow, and Docker. Designed for production fraud detection.*
