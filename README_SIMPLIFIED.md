# Unsupervised Fraud Detection with Autoencoders

This project demonstrates an unsupervised fraud detection system using an autoencoder neural network. The goal is to identify anomalous e-commerce transactions in a synthetic dataset designed to simulate real-world fraud scenarios.

## 🎯 Project Overview

During the exploratory data analysis (EDA), it became clear that the dataset exhibits a largely linear structure, with no single feature or pair of features standing out as clearly indicative of fraud. However, fraudulent behavior is often subtle and nonlinear in nature, making traditional rules-based detection ineffective. This makes an autoencoder model particularly well-suited, as it learns to reconstruct normal transaction patterns and flags deviations without requiring labeled fraud examples.

To improve model performance, I implemented a series of feature engineering strategies, including behavioral flags, temporal markers, and relational indicators. These enhancements helped push the model's AUC-ROC to 75, demonstrating meaningful signal in detecting anomalies. However, I made a conscious decision not to over-tune the features. In a real-world setting, feature engineering would be guided by business knowledge and expert input, not excessive tinkering or blind optimization.

Beyond the model itself, a key focus of this project was the engineering and deployment pipeline. I aimed to create a production-grade file structure, including modularized code, configuration management, and experiment tracking. The application is designed to be deployable to a DigitalOcean droplet, serving real-time fraud inference through an API and user-facing dashboard.

Lastly, this project was also a learning journey into both the fraud detection domain and the broader field of unsupervised learning and anomaly detection. It serves as both a technical artifact and a practical demonstration of how machine learning, software engineering, and domain understanding come together in real-world problem solving.

## 🚀 Key Features

- **No Data Leakage**: Proper train/test separation with leakage-free feature engineering
- **Simplified Architecture**: Clean, maintainable code with clear separation of concerns
- **Three-Stage Optimization**: Broad → Narrow → Final tuning sweep process
- **Production Ready**: Robust error handling, logging, and model persistence
- **High Performance**: Optimized for AUC ROC >= 0.75

- **No Data Leakage**: Proper train/test separation with leakage-free feature engineering
- **Simplified Architecture**: Clean, maintainable code with clear separation of concerns
- **Three-Stage Optimization**: Broad → Narrow → Final tuning sweep process
- **Production Ready**: Robust error handling, logging, and model persistence
- **High Performance**: Optimized for AUC ROC >= 0.75

## 📁 Project Structure

```
├── src/
│   ├── features/
│   │   ├── feature_engineer.py    # Simplified feature engineering
│   │   └── __init__.py
│   ├── models/
│   │   ├── autoencoder.py         # Production autoencoder
│   │   └── __init__.py
│   ├── sweeps/
│   │   ├── sweep_manager.py       # Three-stage sweep system
│   │   └── __init__.py
│   └── utils/
│       ├── data_loader.py         # Data loading and cleaning
│       └── __init__.py
├── tests/
│   └── test_auc_75.py            # AUC 0.75 test
├── configs/
│   └── final_optimized_config.yaml # Model configuration
├── main.py                       # Main pipeline
├── run_sweeps.py                 # Sweep runner
└── README_SIMPLIFIED.md
```

## 📋 Module Overview & Execution Order

### **Core Modules**

#### **1. Data Pipeline (`src/utils/data_loader.py`)**
- **Purpose**: Handles data loading, cleaning, and time-aware splitting
- **Key Functions**: 
  - `load_and_split_data()`: Loads data and performs temporal train/test split
  - `clean_data()`: Applies EDA-driven cleaning rules (capping amounts, clipping ages)
  - `save_cleaned_data()`: Persists cleaned data for reuse

#### **2. Feature Engineering (`src/features/feature_engineer.py`)**
- **Purpose**: Creates 25+ engineered features with leakage-free transformations
- **Key Functions**:
  - `fit_transform()`: Fits on training data, transforms both train/test
  - `_engineer_amount_features()`: Transaction amount transformations
  - `_engineer_temporal_features()`: Time-based fraud indicators
  - `_engineer_risk_flags()`: Domain-specific risk combinations
- **Features**: Amount scaling, temporal markers, customer behavior, risk flags

#### **3. Autoencoder Model (`src/models/autoencoder.py`)**
- **Purpose**: Production-grade autoencoder for anomaly detection
- **Key Functions**:
  - `build_model()`: Constructs neural network architecture
  - `train()`: Trains with early stopping and AUC monitoring
  - `predict_anomaly_scores()`: Generates reconstruction error scores
  - `_calculate_threshold()`: Sets anomaly threshold from normal data
- **Architecture**: Encoder → Latent Space → Decoder with dropout/batch norm

#### **4. Sweep System (`src/sweeps/sweep_manager.py`)**
- **Purpose**: Three-stage hyperparameter optimization
- **Stages**:
  - **Broad**: Tests 5 architectures (small, medium, large, deep, wide)
  - **Narrow**: Focuses on top 3 configurations
  - **Final**: Fine-tunes best configuration
- **Key Functions**: `run_complete_sweep()`, `run_broad_sweep()`, etc.

### **Execution Scripts**

#### **5. Main Pipeline (`main.py`)**
- **Purpose**: Unified entry point for all operations
- **Modes**:
  - `--mode train`: Train model with optimized configuration
  - `--mode predict`: Load model and make predictions
  - `--mode test`: Run AUC 0.75 test
- **Features**: Automatic data cleaning, model persistence, logging

#### **6. Sweep Runner (`run_sweeps.py`)**
- **Purpose**: Orchestrates the three-stage optimization process
- **Stages**: `--stage broad`, `--stage narrow`, `--stage final`, `--stage all`
- **Features**: Automatic data preparation, result tracking, best model selection

#### **7. Testing (`tests/test_auc_75.py`)**
- **Purpose**: Validates system performance and quality
- **Tests**:
  - AUC ROC >= 0.75 requirement
  - Feature engineering quality
  - Model persistence functionality
  - Prediction consistency

### **🔄 Recommended Execution Order**

#### **For First-Time Setup:**
```bash
# 1. Test the system (validates everything works)
python main.py --mode test

# 2. Train the model with optimized configuration
python main.py --mode train

# 3. Make predictions on new data
python main.py --mode predict
```

#### **For Hyperparameter Optimization:**
```bash
# 1. Run complete three-stage sweep
python run_sweeps.py --stage all

# 2. Train final model with best configuration
python main.py --mode train

# 3. Validate performance
python main.py --mode test
```

#### **For Development/Experimentation:**
```bash
# 1. Run individual sweep stages
python run_sweeps.py --stage broad    # Find promising architectures
python run_sweeps.py --stage narrow   # Optimize hyperparameters
python run_sweeps.py --stage final    # Fine-tune best model

# 2. Test changes
python tests/test_auc_75.py

# 3. Train and validate
python main.py --mode train
python main.py --mode test
```

### **📊 Data Flow**

```
Raw Data → Data Loader → Feature Engineer → Autoencoder → Predictions
    ↓           ↓              ↓              ↓            ↓
  Cleaning   Splitting    Transformations   Training   Anomaly Scores
    ↓           ↓              ↓              ↓            ↓
  Capped     Train/Test    Leakage-Free    Threshold   Fraud Labels
  Values     Temporal      Fitted Only     Calculated   Generated
```

## 🚀 Quick Start

### 1. Train Model
```bash
python main.py --mode train
```

### 2. Run Sweeps (Three-Stage Optimization)
```bash
python run_sweeps.py --stage all
```

### 3. Test AUC Performance
```bash
python main.py --mode test
```

### 4. Make Predictions
```bash
python main.py --mode predict
```

## 🔧 Configuration

### Model Configuration
```python
model_config = {
    'latent_dim': 24,              # Latent space dimension
    'hidden_dims': [256, 128, 64, 32],  # Hidden layer dimensions
    'learning_rate': 0.0003,       # Learning rate
    'batch_size': 64,              # Batch size
    'epochs': 100,                 # Training epochs
    'dropout_rate': 0.2,           # Dropout rate
    'threshold_percentile': 95,    # Anomaly threshold percentile
    'early_stopping': True,        # Enable early stopping
    'patience': 15,                # Early stopping patience
    'reduce_lr': True              # Enable learning rate reduction
}
```

## 📊 Feature Engineering

The system includes 25+ engineered features based on EDA insights:

### Transaction Features
- `amount_log`: Log-transformed transaction amount
- `amount_per_item`: Amount per item ratio
- `amount_scaled`: Robust-scaled amount
- `high_amount_95/99`: High amount flags

### Temporal Features
- `hour`: Transaction hour
- `is_late_night`: Late night flag (11 PM - 6 AM)
- `is_business_hours`: Business hours flag (9 AM - 5 PM)

### Customer Features
- `age_group_encoded`: Age group encoding
- `account_age_days_log`: Log-transformed account age
- `new_account`: New account flag (≤7 days)
- `established_account`: Established account flag (>30 days)
- `location_freq`: Location frequency encoding

### Categorical Features
- `payment_method_encoded`: Payment method encoding
- `product_category_encoded`: Product category encoding
- `device_used_encoded`: Device encoding

### Interaction Features
- `amount_quantity_interaction`: Amount × Quantity
- `age_account_interaction`: Age × Account age
- `amount_hour_interaction`: Amount × Hour

### Risk Flags
- `high_quantity`: High quantity flag
- `young_customer`: Young customer flag (≤18)
- `high_risk_combination`: High amount + late night
- `new_account_high_amount`: New account + high amount

## 🔄 Three-Stage Sweep Process

### Stage 1: Broad Sweep
- Tests 5 different architectures (small, medium, large, deep, wide)
- Identifies promising configurations
- Quick evaluation with 50 epochs

### Stage 2: Narrow Sweep
- Focuses on top 3 configurations from Stage 1
- Tests learning rate, batch size, and dropout variations
- More thorough evaluation

### Stage 3: Final Tuning
- Fine-tunes the best configuration from Stage 2
- Optimizes learning rate, threshold, and epochs
- Final performance optimization

## 🧪 Testing

### AUC 0.75 Test
```bash
python tests/test_auc_75.py
```

The test verifies:
- Model achieves AUC ROC >= 0.75
- Feature engineering quality
- Model persistence functionality
- Prediction consistency

## 📈 Performance

Expected performance with optimized configuration:
- **AUC ROC**: ≥ 0.75
- **Training Time**: ~5-10 minutes
- **Memory Usage**: ~2-4 GB
- **Prediction Speed**: ~1000 transactions/second

## 🛠️ Development

### Adding New Features
1. Modify `src/features/feature_engineer.py`
2. Add feature to `get_feature_names()` method
3. Update tests if needed

### Modifying Model Architecture
1. Modify `src/models/autoencoder.py`
2. Update `build_model()` method
3. Test with sweep process

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_auc_75.py
```

## 🚨 Data Leakage Prevention

The system prevents data leakage through:

1. **Immediate Data Splitting**: Data split before any feature engineering
2. **Leakage-Free Feature Engineering**: All transformations fitted on training data only
3. **Proper Scaler Fitting**: Scalers fitted on training data only
4. **Threshold Calculation**: Thresholds calculated from training data only

## 📝 Logging

Comprehensive logging throughout the pipeline:
- Data loading and cleaning
- Feature engineering steps
- Model training progress
- Performance metrics
- Error handling

## 🔒 Model Persistence

Models are saved with:
- Neural network weights (`_model.h5`)
- Scaler parameters (`_scaler.pkl`)
- Threshold value (`_threshold.pkl`)
- Feature engineering objects (`_features.pkl`)

## 🎯 Best Practices

- **Time-Aware Splitting**: Respects temporal order of transactions
- **Robust Scaling**: Handles outliers gracefully
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Optimizes convergence
- **Comprehensive Testing**: Ensures quality and performance

## 🚀 Production Deployment

The system is production-ready with:
- Error handling and logging
- Model persistence and loading
- Configurable parameters
- Performance monitoring
- Clean, maintainable code

## 📊 Monitoring

Monitor key metrics:
- AUC ROC performance
- Training convergence
- Prediction distribution
- Model drift detection
- Feature importance

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or model size
2. **Slow Training**: Reduce epochs or use GPU
3. **Poor Performance**: Run sweep process to optimize hyperparameters
4. **Data Issues**: Check data quality and preprocessing

### Performance Tuning

1. Run broad sweep to find good architecture
2. Run narrow sweep to optimize hyperparameters
3. Run final tuning for best performance
4. Monitor and adjust based on results 


Guidelines to follow:
# 🧠 Unsupervised Fraud Detection with Autoencoders

A production-grade anomaly detection pipeline using an autoencoder to flag fraudulent transactions. Built for reproducibility, clean config management, no data leakage, and experiment tracking.

---

## ✅ Summary of Best Practices

- **Leakage-Free:** Train/test split before all transforms. Test only uses train-fitted scalers.
- **Config-Driven:** All logic controlled via YAML files. No hardcoded values.
- **Smart Sweeps:** 3-stage hyperparameter tuning with auto-promotion of best config if AUC improves.
- **W&B Logging:** Clean, minimal logging of key metrics (AUC, loss, threshold, LR).
- **Full Reproducibility:** Global `seed` from config sets Python, NumPy, and ML framework.
- **Deterministic Operations:** GPU ops forced to be deterministic (e.g. cuDNN).
- **Tests Included:** Validate leakage, AUC, prediction consistency, and config integrity.

---

## 🔧 Config Structure

Config values are stored in:

```
configs/
├── sweep_broad.yaml
├── sweep_narrow.yaml       ← auto-filled by stage 1
├── sweep_final.yaml        ← auto-filled by stage 2
└── final_optimized_config.yaml  ← promoted if AUC improves
```

Sample:
```yaml
seed: 42
model:
  latent_dim: 24
  hidden_dims: [256, 128, 64]
training:
  batch_size: 64
  learning_rate: 0.0003
  epochs: 100
```

---

## ⚙️ Sweep Logic (Auto-Promotion)

| Stage   | What It Does                        | Output Config                   |
|---------|-------------------------------------|----------------------------------|
| Broad   | Test 5 architectures                | → `sweep_narrow.yaml`           |
| Narrow  | Tune top 3 from broad               | → `sweep_final.yaml`            |
| Final   | Fine-tune best config from narrow   | → `final_optimized_config.yaml` (if AUC improves)

Each script prints the **exact command** to run next.

---

## 🧬 Reproducibility Enforcement

- `seed` from config is applied to:
  - `random.seed(seed)`
  - `np.random.seed(seed)`
  - `torch.manual_seed(seed)` or `tf.random.set_seed(seed)`
- Determinism enabled:
  - PyTorch: `torch.use_deterministic_algorithms(True)`
  - TensorFlow: `tf.config.experimental.enable_op_determinism()`
- Ensures identical results across runs with same config + seed.

---

## 🧪 Test Suite

Run with:
```bash
pytest tests/
```

| File                        | Purpose                              |
|-----------------------------|--------------------------------------|
| `test_no_data_leak.py`      | ✅ Ensures no test leakage into train |
| `test_auc_75.py`            | ✅ AUC must meet or exceed 0.75       |
| `test_config_consistency.py`| ✅ Config keys are present and valid  |
| `test_model_reproducibility.py` | ✅ Same seed = same model results |
| `test_prediction_consistency.py` | ✅ Saved model = same outputs   |

---

## 📊 W&B Logging (Clean)

- Metrics: `auc_roc`, `train_loss`, `val_loss`, `threshold`, `lr`, `epoch`
- Grouping: Based on sweep phase (`sweep_broad`, `sweep_final`, etc.)
- Summary: Logs best AUC, config hash, and promoted config info
- No clutter: Avoids `wandb.watch()` and per-layer dumps

---

## 🛠️ Key Commands

```bash
# Step 1: Broad sweep
python run_sweeps.py --stage broad --config configs/sweep_broad.yaml

# Step 2: Narrow sweep
python run_sweeps.py --stage narrow --config configs/sweep_narrow.yaml

# Step 3: Final sweep
python run_sweeps.py --stage final --config configs/sweep_final.yaml

# Step 4: Train best config
python main.py --mode train --config configs/final_optimized_config.yaml

# Step 5: Test model
python main.py --mode test --config configs/final_optimized_config.yaml

# Step 6: Predict on new data
python main.py --mode predict --config configs/final_optimized_config.yaml
```

---

## 📂 Project Layout

```
src/
├── features/               # Feature engineering
├── models/                 # Autoencoder model
├── utils/                  # Data loading, splitting
├── sweeps/                 # Sweep logic + scoring
├── config_loader.py        # YAML + validation
tests/                      # Full test suite
configs/                    # YAML config files
main.py                     # Entry point
run_sweeps.py               # Sweep controller
```

---

## ✅ Summary

> This pipeline is designed for safe, reproducible ML experimentation:
> - No data leakage
> - Deterministic training
> - Config-controlled experimentation
> - Smart sweep auto-promotion
> - Minimal logging, maximal clarity
