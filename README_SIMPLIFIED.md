# Simplified Fraud Detection Pipeline

A clean, production-grade fraud detection system using autoencoders with proper data leakage prevention.

## 🎯 Key Features

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
├── main.py                       # Main pipeline
├── run_sweeps.py                 # Sweep runner
└── README_SIMPLIFIED.md
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