# Fraud Detection Autoencoder

A comprehensive fraud detection system using autoencoders for anomaly detection on the IEEE-CIS Fraud Detection dataset.

## Overview

This project implements an autoencoder-based anomaly detection system for identifying fraudulent transactions. The autoencoder is trained only on legitimate transactions and learns to reconstruct them with minimal error. Fraudulent transactions, being anomalies, will have higher reconstruction errors, allowing us to detect them.

## Features

- **Modular Design**: Clean, maintainable code structure
- **IEEE-CIS Dataset**: Uses the comprehensive IEEE-CIS Fraud Detection dataset with interpretable features
- **Autoencoder Model**: PyTorch-based autoencoder with configurable architecture
- **Comprehensive Evaluation**: Multiple evaluation metrics and visualizations
- **Production Ready**: Includes model saving/loading and preprocessing pipeline

## Project Structure

```
auto-encoder-project/
├── data/
│   └── raw/                    # Raw dataset files
│       ├── train_transaction.csv
│       ├── train_identity.csv
│       ├── test_transaction.csv
│       └── test_identity.csv
├── src/
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── autoencoder.py          # Autoencoder model and trainer
│   └── evaluator.py            # Model evaluation utilities
├── results/                    # Generated results and plots
├── models/                     # Saved models
├── notebooks/
│   └── notebook2               # Original working notebook
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd auto-encoder-project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Kaggle credentials**:
   - Create a Kaggle account and download your API credentials
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

## Usage

### Quick Start

Run the complete pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess the IEEE-CIS dataset
2. Train an autoencoder on legitimate transactions
3. Evaluate the model's fraud detection performance
4. Save results and visualizations to `results/`
5. Save the trained model to `models/`

### Data Loading

The `FraudDataLoader` class handles:
- Loading transaction and identity data
- Merging datasets
- Cleaning and preprocessing
- Feature encoding
- Data scaling and splitting

### Model Training

The `AutoencoderTrainer` class provides:
- Configurable autoencoder architecture
- Training with progress monitoring
- Anomaly detection using reconstruction error
- Threshold calculation

### Evaluation

The `FraudEvaluator` class generates:
- Classification reports
- Confusion matrices
- ROC and PR curves
- Error distribution plots
- Performance metrics

## Model Architecture

The autoencoder uses:
- **Encoder**: Input → 64 → 32 (hidden layers)
- **Decoder**: 32 → 64 → Input
- **Activation**: ReLU
- **Loss**: Mean Squared Error
- **Optimizer**: Adam

## Results

The pipeline generates:
- `results/metrics.json`: Performance metrics
- `results/training_losses.npy`: Training loss history
- `results/confusion_matrix.png`: Confusion matrix plot
- `results/error_distribution.png`: Reconstruction error distribution
- `results/roc_curve.png`: ROC curve
- `results/pr_curve.png`: Precision-Recall curve
- `models/autoencoder_fraud_detection.pth`: Saved model

## Configuration

Key parameters can be modified in `main.py`:
- `hidden_dims`: Autoencoder architecture
- `epochs`: Training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `percentile`: Threshold percentile for anomaly detection

## Dataset

The IEEE-CIS Fraud Detection dataset contains:
- **Transaction data**: Amount, card type, merchant info, etc.
- **Identity data**: Device info, browser, OS, etc.
- **Target**: `isFraud` (0=legitimate, 1=fraudulent)

Features are interpretable and include:
- Transaction amount and type
- Card information
- Merchant details
- Device and browser information
- Geographic location data

## Performance

Typical performance metrics:
- **ROC-AUC**: ~0.85-0.90
- **Precision**: ~0.70-0.80
- **Recall**: ~0.60-0.75
- **F1-Score**: ~0.65-0.75

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IEEE-CIS for providing the fraud detection dataset
- Kaggle for hosting the competition
- PyTorch team for the deep learning framework 