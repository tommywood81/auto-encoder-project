# Autoencoder Fraud Detection System

A comprehensive, production-ready fraud detection system using unsupervised autoencoders with advanced feature engineering and explainable AI capabilities.

## Overview

This system demonstrates how to build a robust fraud detection pipeline that can identify anomalous transactions without requiring labeled fraud data. It combines:

- **Unsupervised Learning**: Autoencoder-based anomaly detection
- **Advanced Feature Engineering**: 30+ engineered features using multiple strategies
- **Explainable AI**: Random Forest explainer for model interpretability
- **Production Deployment**: Docker containerization and API endpoints
- **Interactive Dashboard**: Real-time fraud detection with 3D visualizations

## Key Features

### Unsupervised Fraud Detection
- **Autoencoder Architecture**: Compresses transaction data to detect anomalies
- **No Label Requirements**: Works without historical fraud labels
- **Real-time Scoring**: Processes transactions in milliseconds
- **Adaptive Thresholds**: Dynamic sensitivity adjustment

### Advanced Feature Engineering
- **Multiple Strategies**: 8 different feature engineering approaches
- **Extensible Framework**: Easy to add new feature strategies
- **Performance Optimization**: Efficient feature generation pipeline
- **Comprehensive Coverage**: Temporal, behavioral, demographic, and interaction features

### Explainable AI
- **Random Forest Explainer**: Provides feature importance for each prediction
- **Black Box Interpretation**: Makes autoencoder decisions interpretable
- **Business Insights**: Clear explanations for fraud flags
- **Feature Analysis**: Identifies which features contribute to fraud detection

### Production Ready
- **Docker Deployment**: Containerized application with health checks
- **REST API**: Standardized endpoints for integration
- **Interactive Dashboard**: Real-time fraud detection interface
- **Scalable Architecture**: Handles thousands of transactions

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│ Feature Factory  │───▶│ Autoencoder     │
│   (15 columns)  │    │ (30+ features)   │    │ (16 latent)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ RF Explainer     │    │ Reconstruction  │
                       │ (Interpretable)  │    │ Error Scoring   │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────────────────────────────┐
                       │         Fraud Detection API            │
                       │      (Real-time predictions)           │
                       └─────────────────────────────────────────┘
```

## Feature Engineering Strategies

The system uses a modular feature factory with 8 different strategies:

### 1. Baseline Numeric
- **Purpose**: Core transaction features
- **Features**: Amount, customer age, account age
- **Strategy**: Direct numeric encoding

### 2. Categorical Encoding
- **Purpose**: Encode categorical variables
- **Features**: Payment method, product category, device type
- **Strategy**: One-hot encoding and label encoding

### 3. Temporal Features
- **Purpose**: Time-based patterns
- **Features**: Hour of day, day of week, time since last transaction
- **Strategy**: Cyclical encoding and time differences

### 4. Rolling Statistics
- **Purpose**: Behavioral patterns over time
- **Features**: Rolling averages, standard deviations
- **Strategy**: Window-based aggregations

### 5. Behavioral Features
- **Purpose**: Customer behavior patterns
- **Features**: Transaction frequency, amount patterns
- **Strategy**: Customer-level aggregations

### 6. Rank Encoding
- **Purpose**: Relative positioning
- **Features**: Amount rank, age rank
- **Strategy**: Percentile-based ranking

### 7. Time Interactions
- **Purpose**: Complex temporal relationships
- **Features**: Amount × hour, frequency × amount
- **Strategy**: Feature interactions

### 8. Demographics
- **Purpose**: Customer segmentation
- **Features**: Age groups, location patterns
- **Strategy**: Binning and grouping

## Model Performance

### Current Results
- **AUC-ROC**: 0.73
- **Dataset**: 22,580 transactions (5% fraud rate)
- **Features**: 30+ engineered features
- **Training**: Unsupervised on clean data

### Performance Context
- **Imbalanced Dataset**: Only 5% fraud cases
- **Unsupervised Learning**: No fraud labels used in training
- **Real-world Applicability**: Handles missing fraud labels
- **Improvement Potential**: Better features and semi-supervised learning

## Quick Start

### Prerequisites
- Python 3.8+
- Docker (for deployment)
- Git

### Installation

```bash
# Clone repository
git clone <repository-url>
cd auto-encoder-project

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Local Development

```bash
# Run the application locally
python app.py

# Access the dashboard
# http://localhost:5000
```

### Docker Deployment

```bash
# Deploy with Docker
python deploy_local.py

# Access the dashboard
# http://localhost:5000
```

## API Usage

### Predict Fraud

```python
import requests

# Single transaction prediction
response = requests.post('http://localhost:5000/api/predict', json={
    'transaction_id': 'TXN_001',
    'amount': 150.00,
    'customer_age': 35,
    'payment_method': 'Credit Card',
    'product_category': 'Electronics',
    'device_used': 'Mobile'
})

result = response.json()
print(f"Fraud Probability: {result['fraud_probability']:.3f}")
print(f"Risk Level: {result['risk_level']}")
```

### Batch Processing

```python
# Multiple transactions
transactions = [
    {'transaction_id': 'TXN_001', 'amount': 150.00, ...},
    {'transaction_id': 'TXN_002', 'amount': 75.50, ...}
]

response = requests.post('http://localhost:5000/api/predict_batch', json={
    'transactions': transactions
})

results = response.json()
for result in results['predictions']:
    print(f"{result['transaction_id']}: {result['risk_level']}")
```

## Dashboard Features

### Business Use Tab
- **Date Selection**: Filter by specific dates
- **Threshold Adjustment**: Dynamic fraud sensitivity (80-100%)
- **Real-time Metrics**: Live fraud detection statistics
- **Transaction Table**: Complete transaction analysis
- **Feature Importance**: Explainable AI insights

### Data Science Tab
- **3D Latent Space**: Interactive visualization of autoencoder compression
- **Model Performance**: Comprehensive performance metrics
- **Feature Analysis**: Detailed feature importance analysis
- **Model Understanding**: How the autoencoder works

## Development

### Adding New Features

The system is designed to be easily extensible. To add a new feature strategy:

1. **Create Strategy Class**:
```python
from src.feature_factory.base import FeatureEngineer

class MyNewStrategy(FeatureEngineer):
    def generate_features(self, df):
        # Your feature generation logic
        return df
    
    def get_feature_info(self):
        return {
            'name': 'my_new_strategy',
            'description': 'Description of what this strategy does',
            'features_added': ['feature1', 'feature2']
        }
```

2. **Register in Factory**:
```python
# In src/feature_factory/factory.py
from .strategies.my_new_strategy import MyNewStrategy

FEATURE_STRATEGIES = {
    'my_new_strategy': MyNewStrategy,
    # ... other strategies
}
```

3. **Test and Validate**:
```bash
python -m pytest tests/test_feature_factory.py
```

### Configuration

The system uses YAML configuration files for easy customization:

```yaml
# configs/baseline.yaml
model:
  latent_dim: 16
  epochs: 50
  batch_size: 32

features:
  strategies:
    - baseline_numeric
    - categorical
    - temporal

threshold:
  percentile: 97
```

## Testing

### Run All Tests
```bash
python -m pytest tests/
```

### Run Specific Tests
```bash
python -m pytest tests/test_feature_factory.py
python -m pytest tests/test_models.py
```

### Test Coverage
```bash
python -m pytest --cov=src tests/
```

## Deployment

### Docker Deployment
```bash
# Build and run
python deploy_local.py

# Stop deployment
python deploy_local.py --stop
```

### Production Deployment
```bash
# Deploy to production server
python deploy_pipeline.py --config production.yaml
```

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/
black src/

# Run tests
python -m pytest tests/
```

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to functions
- Write comprehensive docstrings
- Include unit tests for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the example notebooks in `notebooks/`

## Acknowledgments

- Dataset: [Fraudulent E-commerce Transaction Data](https://www.kaggle.com/datasets/arhamrumi/fraudulent-ecommerce-transaction-data)
- Libraries: TensorFlow, Scikit-learn, Pandas, NumPy
- Visualization: Plotly, Matplotlib, Seaborn 