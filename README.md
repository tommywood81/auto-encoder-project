# 🧠 Unsupervised Fraud Detection with Autoencoders

A production-grade anomaly detection pipeline using autoencoders to flag fraudulent e-commerce transactions. Built for reproducibility, clean config management, no data leakage, and experiment tracking.

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [⚙️ Configuration](#️-configuration)
- [🔄 Three-Stage Optimization](#-three-stage-optimization)
- [🧪 Testing](#-testing)
- [📊 Features](#-features)
- [🛠️ Development](#️-development)
- [🔧 Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

This project demonstrates unsupervised fraud detection using autoencoders on synthetic e-commerce data. The dataset exhibits linear structure with subtle fraud patterns, making autoencoders ideal for detecting anomalies without labeled examples.

**Key Achievements:**
- ✅ **AUC ROC ≥ 0.75** target achieved
- ✅ **No Data Leakage** - proper train/test separation
- ✅ **Config-Driven** - all logic via YAML files
- ✅ **Reproducible** - deterministic operations
- ✅ **Production-Ready** - error handling, logging, persistence

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the System
```bash
python main.py --mode test
```

### 3. Train Model
```bash
python main.py --mode train
```

### 4. Make Predictions
```bash
python main.py --mode predict
```

### 5. Run Complete Optimization
```bash
python run_sweeps.py --stage all
```

---

## 📁 Project Structure

```
├── src/
│   ├── config_loader.py        # YAML + validation
│   ├── features/               # Feature engineering
│   ├── models/                 # Autoencoder model
│   ├── sweeps/                 # Sweep logic + scoring
│   └── utils/                  # Data loading, splitting
├── tests/                      # Full test suite
├── configs/                    # YAML config files
├── main.py                     # Entry point
├── run_sweeps.py               # Sweep controller
└── run_tests.py                # Test runner
```

---

## ⚙️ Configuration

### Config Files
```
configs/
├── sweep_broad.yaml           # 5 architectures to test
├── sweep_narrow.yaml          # Top 3 configs (auto-filled)
├── sweep_final.yaml           # Best config (auto-filled)
└── final_optimized_config.yaml # Promoted if AUC improves
```

### Sample Config
```yaml
seed: 42
model:
  latent_dim: 24
  hidden_dims: [256, 128, 64, 32]
  dropout_rate: 0.2
training:
  batch_size: 64
  learning_rate: 0.0003
  epochs: 100
  early_stopping: true
  patience: 15
features:
  threshold_percentile: 95
  use_amount_features: true
  use_temporal_features: true
  use_customer_features: true
  use_risk_flags: true
```

---

## 🔄 Three-Stage Optimization

### Stage 1: Broad Sweep
```bash
python run_sweeps.py --stage broad
```
- Tests 5 architectures (small, medium, large, deep, wide)
- Quick evaluation (50 epochs)
- Outputs: `sweep_narrow.yaml`

### Stage 2: Narrow Sweep
```bash
python run_sweeps.py --stage narrow
```
- Focuses on top 3 configurations
- Tests hyperparameter variations
- Outputs: `sweep_final.yaml`

### Stage 3: Final Tuning
```bash
python run_sweeps.py --stage final
```
- Fine-tunes best configuration
- Auto-promotes if AUC improves
- Outputs: `final_optimized_config.yaml`

### Complete Process
```bash
python run_sweeps.py --stage all
```

---

## 🧪 Testing

### Run All Tests
```bash
python run_tests.py
```

### Individual Tests
```bash
# Data leakage prevention
python tests/test_no_data_leak.py

# AUC 0.75 requirement
python tests/test_auc_75.py

# Config validation
python tests/test_config_consistency.py

# Model reproducibility
python tests/test_model_reproducibility.py

# Prediction consistency
python tests/test_prediction_consistency.py
```

---

## 📊 Features

### Engineered Features (25+)
- **Transaction**: `amount_log`, `amount_per_item`, `amount_scaled`, `high_amount_95/99`
- **Temporal**: `hour`, `is_late_night`, `is_business_hours`
- **Customer**: `age_group_encoded`, `account_age_days_log`, `new_account`, `location_freq`
- **Categorical**: `payment_method_encoded`, `product_category_encoded`, `device_used_encoded`
- **Interactions**: `amount_quantity_interaction`, `age_account_interaction`, `amount_hour_interaction`
- **Risk Flags**: `high_quantity`, `young_customer`, `high_risk_combination`, `new_account_high_amount`

### Model Architecture
- **Encoder**: Dense layers with BatchNorm + Dropout
- **Latent Space**: Configurable dimension (8-32)
- **Decoder**: Symmetric to encoder
- **Training**: Early stopping, learning rate scheduling, AUC monitoring

---

## 🛠️ Development

### Adding Features
1. Modify `src/features/feature_engineer.py`
2. Add to `get_feature_names()` method
3. Update tests if needed

### Modifying Model
1. Modify `src/models/autoencoder.py`
2. Update `build_model()` method
3. Test with sweep process

### Key Commands
```bash
# Train with specific config
python main.py --mode train --config configs/final_optimized_config.yaml

# Run individual sweep stages
python run_sweeps.py --stage broad --config configs/sweep_broad.yaml

# Test specific functionality
python tests/test_auc_75.py
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Memory errors | Reduce `batch_size` or model size |
| Slow training | Reduce `epochs` or use GPU |
| Poor AUC | Run sweep process to optimize |
| Data errors | Check data quality and preprocessing |

### Performance Tuning
1. Run broad sweep → find good architecture
2. Run narrow sweep → optimize hyperparameters  
3. Run final tuning → fine-tune best config
4. Monitor and adjust based on results

### Expected Performance
- **AUC ROC**: ≥ 0.75
- **Training Time**: ~5-10 minutes
- **Memory Usage**: ~2-4 GB
- **Prediction Speed**: ~1000 transactions/second

---

## 🎯 Best Practices

- **✅ Leakage-Free**: Train/test split before all transforms
- **✅ Config-Driven**: All logic via YAML, no hardcoded values
- **✅ Reproducible**: Global seed, deterministic operations
- **✅ Smart Sweeps**: Auto-promotion of best config
- **✅ Clean Logging**: Minimal W&B metrics, no clutter
- **✅ Comprehensive Testing**: 5 test files covering all aspects

---

## 📈 Monitoring

### Key Metrics
- AUC ROC performance
- Training convergence
- Prediction distribution
- Model drift detection
- Feature importance

### W&B Integration
- Clean, minimal logging
- Metrics: `auc_roc`, `train_loss`, `val_loss`, `threshold`, `lr`
- Grouping by sweep phase
- No per-layer dumps

---

> **This pipeline is designed for safe, reproducible ML experimentation with no data leakage, deterministic training, config-controlled experimentation, and smart sweep auto-promotion.** 