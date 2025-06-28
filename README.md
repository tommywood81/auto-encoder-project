# E-commerce Fraud Detection with Autoencoders

> **Can we catch fraudsters using only normal transaction patterns?** 

This project explores how autoencoders can detect fraudulent e-commerce transactions by learning what "normal" looks like. Instead of trying to identify fraud directly, we teach the model to recognize legitimate patterns—anything that doesn't fit becomes a potential red flag.

**Current Performance**: Our best model achieves an **AUC ROC of 0.6511** using a combination of behavioral, temporal, and demographic features.

![Fraud Distribution](./docs/fraud_distribution.png)

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Results](#-results)
- [Contributing](#-contributing)
- [Tests](#-tests)
- [Credits](#-credits)
- [Contact](#-contact)
- [License](#-license)

## ✨ Features

### 🎯 Core Capabilities
- **Unsupervised Learning**: Detects fraud without labeled fraud examples
- **Production-Ready Pipeline**: Modular, configurable, and scalable architecture
- **Multiple Feature Strategies**: Test different approaches systematically
- **Automated Experimentation**: Feature sweep pipeline for performance comparison

### 🏗️ Technical Features
- **Strategy Pattern**: Flexible feature engineering with easy extensibility
- **Configuration Management**: Type-safe, version-controlled experiment settings
- **Modular Design**: Isolated components for data cleaning, feature engineering, and model training
- **Comprehensive Logging**: Full pipeline monitoring and debugging support

### 📊 Feature Engineering Strategies
- **Baseline**: Core transaction features (9 features)
- **Temporal**: Time-based patterns and night-time detection
- **Behavioural**: Purchase behavior analysis and amount-per-item calculations
- **Demographic Risk**: Age-based risk scoring
- **Combined**: All unique features from all strategies (12 features)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd auto-encoder-project

# 2. Create and activate virtual environment
python -m venv env
.\env\Scripts\Activate.ps1  # On Windows
source env/bin/activate     # On Unix/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies
The project requires the following key packages:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities
- `tensorflow` - Deep learning framework
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization

## 💻 Usage

### Quick Start

```bash
# Run with the best-performing strategy
python run_pipeline.py --strategy combined

# See all available strategies
python run_pipeline.py --list-strategies

# Run the full feature sweep (tests all strategies)
python sweep_features.py
```

### Available Strategies

| Strategy | Description | Features |
|----------|-------------|----------|
| `baseline` | Core transaction features only | 9 features |
| `temporal` | Basic features + temporal patterns | 10 features |
| `behavioural` | Core features + amount per item | 10 features |
| `demographic_risk` | Core features + age risk scores | 10 features |
| `combined` | All unique features from all strategies | 12 features |

### Example Output

```
FEATURE SWEEP RESULTS
================================================================================
Strategy             Status     ROC AUC    Notes                          
--------------------------------------------------------------------------------
combined             ✅ SUCCESS 0.6511     Success                        
behavioural          ✅ SUCCESS 0.6235     Success                        
demographic_risk     ✅ SUCCESS 0.6164     Success                        
temporal             ✅ SUCCESS 0.6119     Success                        
baseline             ✅ SUCCESS 0.5954     Success                        

🏆 BEST PERFORMING STRATEGY: combined
   ROC AUC: 0.6511
   Improvement over baseline: +9.36%
```

## 🏛️ Architecture

### System Overview

The project follows a modular, production-ready architecture designed for scalability and maintainability:

```
src/
├── config.py          # Configuration management
├── data/              # Data cleaning and processing
├── feature_factory/   # Strategy pattern for feature engineering
├── models/            # Autoencoder implementation
└── evaluation/        # Model evaluation and metrics
```

### Key Design Patterns

1. **Strategy Pattern**: Feature engineering strategies
2. **Factory Pattern**: Configuration and feature strategy creation
3. **Pipeline Pattern**: Sequential data processing steps
4. **Configuration Pattern**: Centralized settings management
5. **Observer Pattern**: Logging and monitoring throughout pipeline

### Configuration Management

The heart of the system is a sophisticated configuration system that manages everything from data paths to model hyperparameters:

```python
@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    name: str
    description: str
    feature_strategy: str
    data: DataConfig
    model: ModelConfig
    features: FeatureConfig
    
    @classmethod
    def get_config(cls, strategy: str) -> 'PipelineConfig':
        """Get configuration by strategy name."""
        if strategy == "baseline":
            return cls.get_baseline_config()
        elif strategy == "temporal":
            return cls.get_temporal_config()
        elif strategy == "behavioural":
            return cls.get_behavioural_config()
        elif strategy == "demographic_risk":
            return cls.get_demographic_risk_config()
        elif strategy == "combined":
            return cls.get_combined_config()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
```

**Why This Design?** I wanted to make experiments reproducible and configurable. Each strategy has its own factory method, making it easy to version-control different experimental setups.

### Feature Factory Pattern

The feature engineering system uses the **Strategy Pattern** to make it easy to experiment with different feature combinations:

```python
class FeatureEngineer(ABC):
    """Abstract base class for feature engineering strategies."""
    
    @abstractmethod
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for the given dataset."""
        pass
    
    @abstractmethod
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the features generated."""
        pass


class FeatureFactory:
    """Factory for creating feature engineering strategies."""
    
    STRATEGY_CLASSES = {
        "baseline": BaselineFeatures,
        "temporal": TemporalFeatures,
        "behavioural": BehaviouralFeatures,
        "demographic_risk": DemographicRiskFeatures,
        "combined": CombinedFeatures
    }
    
    @classmethod
    def create(cls, strategy_name: str) -> FeatureEngineer:
        """Create a feature engineering strategy."""
        return cls.STRATEGY_CLASSES[strategy_name]()
```

**Why This Pattern?** It makes adding new feature strategies trivial. Want to test a new behavioral feature? Just implement the interface and add it to the factory.

### Pipeline Workflow

The main pipeline (`run_pipeline.py`) orchestrates the entire fraud detection workflow:

```python
def run_pipeline(strategy: str):
    """Run the complete fraud detection pipeline with given strategy."""
    
    # 1. Load configuration
    config = PipelineConfig.get_config(strategy)
    
    # 2. Data cleaning
    cleaner = DataCleaner(config)
    df_cleaned = cleaner.clean_data(save_output=True)
    
    # 3. Feature engineering
    feature_engineer = FeatureFactory.create(config.feature_strategy)
    df_features = feature_engineer.generate_features(df_cleaned)
    
    # 4. Model training
    autoencoder = BaselineAutoencoder(config)
    results = autoencoder.train()
    
    return results
```

Each step is isolated and can be tested independently, making the system robust and maintainable.

## 📈 Results

### Dataset Overview

We're working with **23,634 e-commerce transactions** where about 5.17% are fraudulent (1,222 fraudulent vs 22,412 legitimate). This imbalance makes it a perfect candidate for anomaly detection.

![Dataset Summary](./docs/dataset_summary.png)

### Data Quality Decisions

#### Customer Age Filtering
**The Problem**: Some customers had unrealistic ages (like 5-year-olds making purchases).
**The Decision**: Filtered to customers 18+ years old.
**Legal Reasoning**: Users under 18 cannot legally agree to user agreements and terms of service, making their transactions potentially invalid from a legal standpoint.
**Impact**: Focused on adult customers with predictable fraud patterns, removing 1,054 unrealistic transactions. For now, we're dropping under-18 users and will consult with management on how they want to handle this demographic.

![Customer Age Analysis](./docs/customer_age_analysis.png)

#### Transaction Amount Handling
**The Problem**: Transaction amounts were heavily skewed (skewness of 6.7).
**The Decision**: Used log transformation instead of capping.
**Why It Worked**: Capping would break generalizability for new high-value products. Log transformation reduced skewness to -0.228 while preserving the ability to handle new products.

![Transaction Amount Analysis](./docs/transaction_amount_analysis.png)
![Transaction Amount Skewness](./docs/transaction_amount_skewness_analysis.png)

#### Feature Selection Strategy
**The Problem**: Some columns looked useful but were actually noise.
**The Decision**: Removed 5 columns (Transaction ID, Customer ID, IP Address, Shipping/Billing Address) because they contained all unique values.
**Why It Worked**: Reduced noise and improved model convergence.

![Missing Values Analysis](./docs/missing_values.png)

### Feature Sweep Performance

Our systematic testing revealed that combining all feature strategies provides the best performance:

| Strategy | ROC AUC | Improvement | What It Tests |
|----------|---------|-------------|---------------|
| **Combined** | **0.6511** | **+9.36%** | All features together |
| Behavioural | 0.6235 | +4.72% | Purchase behavior patterns |
| Demographic Risk | 0.6164 | +3.52% | Age-based risk scoring |
| Temporal | 0.6119 | +2.77% | Time-based patterns |
| Baseline | 0.5954 | - | Core transaction features |

### Model Performance Visualization

![ROC Curve](./docs/roc_curve.png)
![PR Curve](./docs/pr_curve.png)
![Confusion Matrix](./docs/confusion_matrix.png)

### Key Insights

The results show that:
1. **More features help**: The combined strategy outperforms individual strategies
2. **Behavioral patterns matter**: Amount per item is a strong fraud indicator
3. **Time matters**: Night-time transactions are indeed suspicious
4. **Age matters**: Younger customers show higher fraud risk

### Performance Metrics

- **Best Strategy**: Combined features (ROC AUC: 0.6511)
- **Improvement over Baseline**: +9.36%
- **Training Time**: ~1 minute per strategy
- **Total Features**: 12 unique features

### Temporal Patterns

![Temporal Patterns](./docs/temporal_patterns.png)

### Payment Method Analysis

![Payment Method Analysis](./docs/payment_method_analysis.png)

### Error Distribution

![Error Distribution](./docs/error_distribution.png)

## 🤝 Contributing

This project demonstrates production-ready ML engineering practices. We welcome contributions!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Add new feature strategies, improve models, or enhance documentation
4. **Test your changes**: Ensure all tests pass and the pipeline runs successfully
5. **Submit a pull request**: Describe your changes and their impact

### Areas for Contribution

- **New Feature Strategies**: Implement `FeatureEngineer` interface
- **Model Improvements**: Enhance the autoencoder architecture
- **Evaluation Metrics**: Add new performance measures
- **Documentation**: Improve guides and examples
- **Performance Optimization**: Speed up the pipeline

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

## 🧪 Tests

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_feature_factory.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Test Coverage

The project includes comprehensive tests for:
- Feature engineering strategies
- Configuration management
- Data cleaning pipeline
- Model training and evaluation
- Pipeline orchestration

### Continuous Integration

Tests are automatically run on:
- Pull requests
- Main branch commits
- Release tags

## 👥 Credits

### Dataset
- **Source**: [Fraudulent E-commerce Transaction Data](https://www.kaggle.com/datasets/arhamrumi/fraudulent-ecommerce-transaction-data)
- **License**: CC0 1.0 Universal

### Key Libraries
- **TensorFlow**: Deep learning framework
- **Scikit-learn**: Machine learning utilities
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

### Inspiration
This project was inspired by the need for robust, unsupervised fraud detection systems that can adapt to evolving fraud patterns without requiring labeled fraud examples.

## 📞 Contact

### Get in Touch
- **Issues**: [GitHub Issues](https://github.com/yourusername/auto-encoder-project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/auto-encoder-project/discussions)
- **Email**: your.email@example.com

### Support
For questions about:
- **Installation**: Check the [Installation](#-installation) section
- **Usage**: See the [Usage](#-usage) section and examples
- **Architecture**: Review the [Architecture](#-architecture) documentation
- **Contributing**: Read the [Contributing](#-contributing) guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### License Summary
- **Permissions**: Commercial use, modification, distribution, private use
- **Limitations**: Liability, warranty
- **Conditions**: License and copyright notice

---

*This project demonstrates that with the right architecture and systematic experimentation, unsupervised learning can be surprisingly effective for fraud detection. The modular design makes it easy to extend and improve as new insights emerge.* 