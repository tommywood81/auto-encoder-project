# Autoencoder Experiment Tracking Pipeline
**With W&B integration, modular sweep scripts, and cake-level logic**

---

## What We've Built

We've implemented a complete **autoencoder fraud detection pipeline** with Weights & Biases (W&B) integration for experiment tracking and hyperparameter tuning. The system automatically:

- Tests different **feature strategies** (ingredients)
- Tunes **model hyperparameters** (baking settings)
- Tracks all experiments in W&B
- Makes smart decisions about when to proceed
- Trains the final production model

---

## Project Structure

```
auto-encoder-project/
├── sweep_features_wandb.py      # Feature strategy testing
├── sweep_parameters_wandb.py    # Hyperparameter tuning
├── main_controller.py           # Orchestrates everything
├── train_final_model.py         # Final model training
├── configs/                     # Configuration files
│   ├── baseline.yaml           # Default settings
│   ├── best_features.yaml      # Auto-generated (best features)
│   └── final_config.yaml       # Auto-generated (best hyperparams)
├── models/                      # Trained models
│   ├── final_model.h5          # Production model
│   └── model_info.yaml         # Model metadata
└── src/
    ├── config_loader.py        # YAML config management
    ├── models/                 # Autoencoder implementations
    └── feature_factory/        # Feature engineering strategies
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up W&B
```bash
# Login to W&B (first time only)
wandb login

# Set your entity (username/team)
export WANDB_ENTITY=your-username
```

### 3. Run the Complete Pipeline
```bash
# Run everything automatically
python main_controller.py --entity your-username

# Or run individual steps:
python sweep_features_wandb.py --entity your-username
python sweep_parameters_wandb.py --entity your-username
python train_final_model.py --entity your-username
```

---

## Step-by-Step Pipeline

### Step 1: Feature Sweep (`sweep_features_wandb.py`)

**Goal**: Find the best feature combination for fraud detection.

**What it does**:
- Tests 5 feature strategies: `baseline`, `temporal`, `behavioural`, `demographic_risk`, `combined`
- Uses fixed model settings (10 epochs, basic hyperparameters)
- Tracks ROC AUC for each strategy in W&B
- Saves best strategy to `configs/best_features.yaml`

**Usage**:
```bash
python sweep_features_wandb.py --entity your-username
```

**W&B Tracking**:
- Project: `fraud-detection-autoencoder`
- Tags: `["feature_sweep"]`
- Metrics: `final_auc`, `precision`, `recall`, `confusion_matrix`
- Config: All model parameters + feature strategy

---

### Step 2: Hyperparameter Tuning (`sweep_parameters_wandb.py`)

**Goal**: Optimize model hyperparameters using the best feature strategy.

**Three Stages**:

#### Stage 2.1: Broad Sweep (10 epochs)
- Tests 48 hyperparameter combinations
- Parameters: `latent_dim`, `learning_rate`, `activation_fn`, `batch_size`, `threshold`
- Takes top 10 configurations

#### Stage 2.2: Refined Sweep (15 epochs)
- Tests top 10 configurations from broad sweep
- Longer training to get better estimates

#### Stage 2.3: Final Training (50 epochs + early stopping)
- Tests top 3 configurations from refined sweep
- Full training with early stopping
- Saves best configuration to `configs/final_config.yaml`

**Usage**:
```bash
# Run all stages
python sweep_parameters_wandb.py --entity your-username

# Run specific stage
python sweep_parameters_wandb.py --entity your-username --stage broad
```

**W&B Tracking**:
- Tags: `["param_sweep", "stage_broad/refined/final"]`
- Metrics: All training metrics + model artifacts
- Artifacts: Final trained model

---

### Step 3: Main Controller (`main_controller.py`)

**Goal**: Orchestrate the entire pipeline with smart decision-making.

**What it does**:
1. Runs feature sweep
2. Checks W&B for best AUC
3. Only proceeds to hyperparameter tuning if improvement > baseline threshold
4. Runs final model training
5. Provides experiment summaries

**Usage**:
```bash
# Run complete pipeline
python main_controller.py --entity your-username

# Force hyperparameter tuning (even without improvement)
python main_controller.py --entity your-username --force-hyperparam

# Show experiment summary only
python main_controller.py --entity your-username --summary

# Custom baseline threshold
python main_controller.py --entity your-username --baseline-threshold 0.75
```

**Smart Features**:
- **Performance-aware**: Only tunes if features actually improve performance
- **W&B Integration**: Uses W&B API to check results automatically
- **Resumable**: Can restart from any stage
- **Monitoring**: Real-time progress tracking

---

### Step 4: Final Training (`train_final_model.py`)

**Goal**: Train the production-ready model with best configuration.

**What it does**:
- Loads best configuration from hyperparameter sweep
- Trains model with full dataset (50 epochs + early stopping)
- Saves model to `models/final_model.h5`
- Creates W&B artifact for model versioning
- Saves model metadata to `models/model_info.yaml`

**Usage**:
```bash
python train_final_model.py --entity your-username
```

**Outputs**:
- `models/final_model.h5`: Production model
- `models/model_info.yaml`: Model metadata and configuration
- W&B artifact: Versioned model for deployment

---

## W&B Dashboard

### What You'll See

1. **Project Overview**:
   - All experiment runs organized by tags
   - Performance comparisons across strategies
   - Model versioning and artifacts

2. **Feature Sweep Results**:
   - ROC AUC comparison across feature strategies
   - Confusion matrices for each strategy
   - Training curves and metrics

3. **Hyperparameter Tuning**:
   - Parameter importance analysis
   - Performance heatmaps
   - Best configuration tracking

4. **Final Model**:
   - Production model metrics
   - Model artifacts for deployment
   - Complete training history

### Key Metrics Tracked

- **`final_auc`**: ROC AUC score (primary metric)
- **`precision`**: Precision score
- **`recall`**: Recall score
- **`threshold`**: Anomaly detection threshold
- **`best_epoch`**: Best training epoch
- **`feature_count`**: Number of features used
- **`fraud_ratio`**: Dataset fraud ratio

---

## Configuration Management

### Configuration Files

The system uses YAML configuration files for easy management:

#### `configs/baseline.yaml`
```yaml
model:
  latent_dim: 16
  learning_rate: 0.001
  activation_fn: "relu"
  batch_size: 128
  epochs: 10
  threshold: 95

features:
  strategy: "baseline"

training:
  validation_split: 0.2
  early_stopping_patience: 5
  reduce_lr_patience: 3
  min_delta: 0.001

wandb:
  project: "fraud-detection-autoencoder"
  entity: null  # Set by user
  tags: ["baseline"]
```

#### Auto-generated Files
- `configs/best_features.yaml`: Updated with best feature strategy
- `configs/final_config.yaml`: Updated with best hyperparameters

### Configuration Loading

```python
from src.config_loader import ConfigLoader

loader = ConfigLoader()
config = loader.load_config("baseline")
config = loader.get_wandb_config("baseline", entity="your-username")
```

---

## Advanced Usage

### Custom Hyperparameter Ranges

Edit `sweep_parameters_wandb.py` to modify hyperparameter search space:

```python
hyperparams = {
    'latent_dim': [8, 16, 32, 64],  # Add more options
    'learning_rate': [0.001, 0.0005, 0.0001],  # Add more options
    'activation_fn': ['relu', 'leaky_relu', 'tanh'],  # Add more options
    'batch_size': [32, 64, 128, 256],  # Add more options
    'threshold': [85, 90, 95, 99]  # Add more options
}
```

### Custom Feature Strategies

Add new feature strategies in `src/feature_factory/strategies.py`:

```python
class CustomStrategy(BaseFeatureStrategy):
    def generate_features(self, df):
        # Your custom feature engineering
        return df_features
```

### Custom Metrics

Add custom metrics in any training script:

```python
# In training function
custom_metric = calculate_custom_metric(y_true, y_pred)
wandb.log({"custom_metric": custom_metric})
```

---

## Production Deployment

### Model Deployment

After training, the final model is ready for deployment:

```python
# Load the trained model
from src.models import BaselineAutoencoder
from src.config import PipelineConfig

config = PipelineConfig.get_config("combined")  # or best strategy
autoencoder = BaselineAutoencoder(config)
autoencoder.load_model("models/final_model.h5")

# Make predictions
predictions = autoencoder.predict_anomaly_scores(new_data)
```

### Docker Integration

The trained model works with your existing Docker setup:

```dockerfile
# Copy the final model
COPY models/final_model.h5 /app/models/
COPY models/model_info.yaml /app/models/
```

---

## Monitoring and Maintenance

### Experiment Tracking

- **W&B Dashboard**: Monitor all experiments in real-time
- **Model Versioning**: Track model performance over time
- **Artifact Management**: Version control for models and configs

### Performance Monitoring

- **AUC Tracking**: Monitor model performance degradation
- **Feature Drift**: Track feature distribution changes
- **Threshold Optimization**: Adjust anomaly detection thresholds

### Retraining Pipeline

```bash
# Retrain with new data
python main_controller.py --entity your-username

# Compare with previous models in W&B
python main_controller.py --entity your-username --summary
```

---

## Benefits

### For Data Scientists
- **Automated Experimentation**: No manual config management
- **Reproducible Results**: All experiments tracked in W&B
- **Performance Optimization**: Smart hyperparameter tuning
- **Easy Comparison**: Visual comparison of all strategies

### For Engineers
- **Production Ready**: Final model ready for deployment
- **Version Control**: Model artifacts versioned in W&B
- **Monitoring**: Easy integration with existing monitoring
- **Scalable**: Modular design for easy extension

### For Business
- **Faster Iteration**: Automated pipeline reduces time to production
- **Better Performance**: Systematic optimization improves model quality
- **Transparency**: All experiments tracked and explainable
- **Cost Effective**: Smart decision-making saves computational resources

---

## Troubleshooting

### Common Issues

1. **W&B Login Issues**:
   ```bash
   wandb login
   # Enter your API key from https://wandb.ai/settings
   ```

2. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration Errors**:
   - Check YAML syntax in config files
   - Ensure all required fields are present

4. **Model Loading Issues**:
   - Verify model file exists: `models/final_model.h5`
   - Check model info: `models/model_info.yaml`

### Getting Help

- **W&B Documentation**: https://docs.wandb.ai/
- **Project Issues**: Check GitHub issues
- **Configuration**: Review `configs/` directory examples

---

## Next Steps

1. **Run the Pipeline**: Start with `python main_controller.py --entity your-username`
2. **Explore W&B**: Check your W&B dashboard for results
3. **Customize**: Modify hyperparameters or add new feature strategies
4. **Deploy**: Use the final model in your production environment
5. **Monitor**: Set up ongoing monitoring and retraining

---

**Happy Experimenting!** 